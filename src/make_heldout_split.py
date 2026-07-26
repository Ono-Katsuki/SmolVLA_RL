"""Generate a held-out split based on LIBERO-Plus perturbation categories.

Splits the episode indices of the `lerobot/libero_plus` HF dataset into
train / heldout.

Two ways to obtain categories:
  1. --episode_categories <csv>  (recommended)
     Per-episode labels recovered from RLDS by recover_episode_categories.py
     (data/episode_categories.csv). Leakage-safe, with task/length
     cross-checks.
  2. No CSV (legacy): inferred by language matching against
     task_classification.json. With the public metadata not all episodes can
     be mapped, so this path currently always fails (leakage-prevention
     guard).

Output:
    <output_dir>/train_episodes.json    # training episode index array
    <output_dir>/heldout_episodes.json  # evaluation episode index array
    <output_dir>/split_summary.json     # per-category counts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_HELDOUT = ["camera_pose"]
# The evaluation benchmark (task_classification.json) has 7 categories, but
# only 5 actually occur in the released training data (RLDS): camera_pose /
# language / lighting / sensor_noise / env ("env" is a mix of background and
# layout perturbations).
CATEGORIES = [
    "object_layout",
    "camera_pose",
    "initial_state",
    "language",
    "lighting",
    "background",
    "sensor_noise",
    "env",
]

CATEGORY_ALIASES = {
    "Objects Layout": "object_layout",
    "Camera Viewpoints": "camera_pose",
    "Robot Initial States": "initial_state",
    "Language Instructions": "language",
    "Light Conditions": "lighting",
    "Background Textures": "background",
    "Sensor Noise": "sensor_noise",
}


def load_task_classification(libero_root: Path) -> dict:
    path = libero_root / "libero" / "libero" / "benchmark" / "task_classification.json"
    if not path.exists():
        raise FileNotFoundError(
            f"task_classification.json not found at {path}. "
            "check that the LIBERO-plus repo has been cloned"
        )
    with path.open() as f:
        return json.load(f)


def load_libero_plus_meta(cache_dir: Path) -> dict:
    """Download lerobot/libero_plus meta and read episodes.jsonl"""
    repo_dir = snapshot_download(
        repo_id="lerobot/libero_plus",
        repo_type="dataset",
        allow_patterns=["meta/*"],
        cache_dir=str(cache_dir),
    )
    meta_dir = Path(repo_dir) / "meta"
    episodes_path = meta_dir / "episodes.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"
    if episodes_path.exists() and tasks_path.exists():
        episodes = [json.loads(l) for l in episodes_path.read_text().splitlines() if l.strip()]
        tasks = [json.loads(l) for l in tasks_path.read_text().splitlines() if l.strip()]
    else:
        import pyarrow.parquet as pq

        episode_files = sorted((meta_dir / "episodes").rglob("*.parquet"))
        if not episode_files or not (meta_dir / "tasks.parquet").exists():
            raise FileNotFoundError(f"unsupported LeRobot metadata layout under {meta_dir}")
        episodes = []
        for path in episode_files:
            episodes.extend(pq.read_table(path).to_pylist())
        tasks = pq.read_table(meta_dir / "tasks.parquet").to_pylist()
    return {"episodes": episodes, "tasks": tasks}


def task_to_category(task_name: str, classification: dict) -> str | None:
    """task_classification.json is either {category: {level: [task_id, ...]}} or
    {task_id: {category, level}}. Support both."""
    # Current LIBERO-Plus schema: {suite: [{id, name, category, ...}, ...]}.
    for entries in classification.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and entry.get("name") == task_name:
                return CATEGORY_ALIASES.get(entry.get("category"), entry.get("category"))

    if task_name in classification:
        entry = classification[task_name]
        if isinstance(entry, dict):
            category = entry.get("category")
            return CATEGORY_ALIASES.get(category, category)
        return entry
    for cat, payload in classification.items():
        if not isinstance(payload, dict):
            continue
        for _level, task_list in payload.items():
            if isinstance(task_list, list) and task_name in task_list:
                return cat
    return None


def load_episode_categories(csv_path: Path, meta: dict) -> dict[int, str]:
    """Read the CSV from recover_episode_categories.py, cross-check against meta, and return it."""
    import csv as csv_mod

    rows = list(csv_mod.DictReader(csv_path.open()))
    if len(rows) != len(meta["episodes"]):
        raise RuntimeError(
            f"episode count mismatch: csv={len(rows)} meta={len(meta['episodes'])}"
        )
    task_names_by_index = {t["task_index"]: t["task"] for t in meta["tasks"]}
    mapping: dict[int, str] = {}
    for ep, row in zip(sorted(meta["episodes"], key=lambda e: e["episode_index"]), rows):
        ep_idx = ep["episode_index"]
        if ep_idx != int(row["episode_index"]):
            raise RuntimeError(f"episode_index mismatch at {ep_idx} vs {row['episode_index']}")
        task_ref = ep["tasks"][0] if isinstance(ep.get("tasks"), list) else ep.get("task_index")
        task_name = task_ref if isinstance(task_ref, str) else task_names_by_index.get(task_ref, "")
        if task_name != row["task"] or int(ep["length"]) != int(row["length"]):
            raise RuntimeError(
                f"episode {ep_idx}: meta (task={task_name!r}, len={ep['length']}) != "
                f"csv (task={row['task']!r}, len={row['length']}) — the CSV may be stale"
            )
        mapping[ep_idx] = row["category"]
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--libero_root",
        type=Path,
        default=Path("/content/LIBERO-plus"),
        help="root of the LIBERO-plus repo (must contain task_classification.json)",
    )
    ap.add_argument(
        "--episode_categories",
        type=Path,
        default=None,
        help="per-episode category CSV emitted by recover_episode_categories.py (recommended)",
    )
    ap.add_argument(
        "--heldout_categories",
        nargs="+",
        default=DEFAULT_HELDOUT,
        choices=CATEGORIES,
        help="categories to exclude from training (default: camera_pose)",
    )
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--cache_dir", type=Path, default=Path("/content/hf_cache"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Invalidate the outputs first. If a split left by a previous/legacy run
    # remains in output_dir and this run raises during validation, the stale
    # (possibly leakage-containing) split survives, and train_sft.sh would
    # silently use it after a mere existence check (caught by Codex).
    for stale in ("train_episodes.json", "heldout_episodes.json", "split_summary.json"):
        (args.output_dir / stale).unlink(missing_ok=True)

    meta = load_libero_plus_meta(args.cache_dir)
    task_names_by_index = {t["task_index"]: t["task"] for t in meta["tasks"]}

    episode_categories: dict[int, str] | None = None
    classification: dict | None = None
    if args.episode_categories is not None:
        episode_categories = load_episode_categories(args.episode_categories, meta)
    else:
        classification = load_task_classification(args.libero_root)

    train_eps: list[int] = []
    heldout_eps: list[int] = []
    per_cat_count: dict[str, int] = {c: 0 for c in CATEGORIES}
    unmapped = 0

    for ep in meta["episodes"]:
        ep_idx = ep["episode_index"]
        if episode_categories is not None:
            cat = episode_categories.get(ep_idx)
        else:
            task_ref = ep["tasks"][0] if isinstance(ep.get("tasks"), list) else ep.get("task_index")
            task_name = (
                task_ref if isinstance(task_ref, str) else task_names_by_index.get(task_ref, "")
            )
            cat = task_to_category(task_name, classification)
        if cat is None:
            unmapped += 1
            train_eps.append(ep_idx)
            continue
        per_cat_count[cat] = per_cat_count.get(cat, 0) + 1
        if cat in args.heldout_categories:
            heldout_eps.append(ep_idx)
        else:
            train_eps.append(ep_idx)

    summary = {
        "heldout_categories": args.heldout_categories,
        "category_source": (
            str(args.episode_categories) if episode_categories is not None else "language_matching"
        ),
        "n_train": len(train_eps),
        "n_heldout": len(heldout_eps),
        "n_unmapped": unmapped,
        "per_category": per_cat_count,
    }
    # Write the split files only after passing the leakage guard. They used to
    # be written first, so even when the guard raised, a usable
    # (leakage-containing) train split remained and train_sft.sh would
    # silently train on it after a mere existence check (caught by Codex).
    if unmapped:
        raise RuntimeError(
            f"{unmapped}/{len(meta['episodes'])} episodes could not be mapped to a "
            "LIBERO-Plus perturbation category. The public LeRobot metadata stores "
            "only 40 natural-language tasks, and several perturbation categories share "
            "the same language. Refusing to emit a leakage-prone split."
        )
    if not heldout_eps:
        raise RuntimeError(
            "heldout split is empty; check the episode-to-category metadata before training"
        )
    # Only after all validations pass are the split files written.
    (args.output_dir / "train_episodes.json").write_text(json.dumps(train_eps))
    (args.output_dir / "heldout_episodes.json").write_text(json.dumps(heldout_eps))
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
