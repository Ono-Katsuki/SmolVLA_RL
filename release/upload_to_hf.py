"""Upload the GRPO and SFT checkpoints to the Hugging Face Hub (Colab VM).

Designed to run on the Colab VM where the checkpoints live. Reads the write
token from the HF_TOKEN environment variable and NEVER prints it. Validates the
LeRobot checkpoint layout before touching the network, bundles the model card as
README.md, and refuses to push into an existing repo unless --exist_ok is given.

Default plan (override paths/names with flags):
  grpo: /content/grpo_run8/ckpt_best      -> <user>/smolvla-grpo-libero-spatial
  sft:  /content/outputs/sft_base_heldout/checkpoints/last
                                          -> <user>/smolvla-sft-libero-spatial-heldout

NOTE: train_grpo.py saves checkpoints as iter_XXXXX/ and ckpt_latest/, each
containing a pretrained_model/ subdirectory. There is no automatic "ckpt_best":
pick the best iteration from metrics.jsonl and pass it (or symlink/copy it to
ckpt_best) before running. Both "<dir>" and "<dir>/pretrained_model" are
accepted; the pretrained_model layout is resolved automatically.

Usage:
    # dry run (no network, no token needed) — validates layout and prints plan:
    python release/upload_to_hf.py --user <HF_USERNAME> --dry_run

    # real upload (needs HF_TOKEN with write scope in the environment):
    HF_TOKEN=... python release/upload_to_hf.py --user <HF_USERNAME>

    # single model / custom checkpoint:
    python release/upload_to_hf.py --user <HF_USERNAME> --only grpo \
        --grpo_ckpt /content/grpo_run8/iter_00012 --dry_run
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# LeRobot save_pretrained layout (see src/grpo/train_grpo.py::_save_checkpoint).
REQUIRED_FILES = ("config.json", "model.safetensors")
EXPECTED_FILES = ("policy_preprocessor.json", "policy_postprocessor.json")
# Never upload these even if present next to the model files.
EXCLUDE_PATTERNS = ("optim.pt", "ITER.txt", "*.lock", ".git*")


@dataclass
class ModelSpec:
    key: str
    ckpt: Path
    repo_name: str
    card: Path


def resolve_model_dir(ckpt: Path) -> Path:
    """Accept either the checkpoint dir or its pretrained_model/ subdir."""
    if (ckpt / "pretrained_model").is_dir():
        return ckpt / "pretrained_model"
    return ckpt


def validate_model_dir(model_dir: Path) -> list[str]:
    """Return a list of problems (empty = OK). Warns (stderr) on soft issues."""
    problems: list[str] = []
    if not model_dir.is_dir():
        return [f"not a directory: {model_dir}"]
    for f in REQUIRED_FILES:
        if not (model_dir / f).is_file():
            problems.append(f"missing required file: {model_dir / f}")
    for f in EXPECTED_FILES:
        if not (model_dir / f).is_file():
            print(f"  [warn] expected file not found (continuing): {model_dir / f}",
                  file=sys.stderr)
    st = model_dir / "model.safetensors"
    if st.is_file() and st.stat().st_size < 1_000_000:
        problems.append(f"model.safetensors is suspiciously small "
                        f"({st.stat().st_size} bytes): {st}")
    return problems


def check_card(card: Path) -> list[str]:
    problems: list[str] = []
    if not card.is_file():
        return [f"model card not found: {card}"]
    text = card.read_text()
    if not text.startswith("---"):
        problems.append(f"model card lacks YAML frontmatter: {card}")
    if "TBD" in text or "FILL" in text:
        print(f"  [warn] {card.name} still contains TBD/FILL placeholders — "
              f"fine for a first upload, update after eval.", file=sys.stderr)
    return problems


def upload(spec: ModelSpec, user: str, token: str, exist_ok: bool,
           private: bool, dry_run: bool) -> None:
    repo_id = f"{user}/{spec.repo_name}"
    model_dir = resolve_model_dir(spec.ckpt)

    print(f"\n=== {spec.key}: {model_dir} -> {repo_id} ===")
    problems = validate_model_dir(model_dir) + check_card(spec.card)
    if problems:
        for p in problems:
            print(f"  [error] {p}", file=sys.stderr)
        raise SystemExit(f"validation failed for '{spec.key}' — nothing uploaded.")

    from fnmatch import fnmatch

    files = sorted(
        p.name for p in model_dir.iterdir() if p.is_file()
        and not any(fnmatch(p.name, pat) for pat in EXCLUDE_PATTERNS)
    )
    print(f"  files: {files}  (excluded: {EXCLUDE_PATTERNS})")
    print(f"  card:  {spec.card}")
    print(f"  repo:  {repo_id} (private={private}, exist_ok={exist_ok})")

    if dry_run:
        print("  [dry_run] validation OK — skipping network operations.")
        return

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo_id, repo_type="model",
                        private=private, exist_ok=exist_ok)
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 409:
            raise SystemExit(
                f"repo {repo_id} already exists; re-run with --exist_ok to "
                f"update it (refusing to overwrite by default).") from e
        raise

    api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(model_dir),
        ignore_patterns=list(EXCLUDE_PATTERNS),
        commit_message=f"upload {spec.key} checkpoint from SmolVLA_RL",
    )
    api.upload_file(
        repo_id=repo_id, repo_type="model", path_or_fileobj=str(spec.card),
        path_in_repo="README.md",
        commit_message=f"model card for {spec.key}",
    )
    print(f"  uploaded -> https://huggingface.co/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True,
                    help="HF username or org (repos are created under it)")
    ap.add_argument("--grpo_ckpt", type=Path,
                    default=Path("/content/grpo_run8/ckpt_best"))
    ap.add_argument("--sft_ckpt", type=Path,
                    default=Path("/content/outputs/sft_base_heldout/checkpoints/last"))
    ap.add_argument("--grpo_repo", default="smolvla-grpo-libero-spatial")
    ap.add_argument("--sft_repo", default="smolvla-sft-libero-spatial-heldout")
    ap.add_argument("--only", choices=["grpo", "sft"], default=None,
                    help="upload just one of the two models")
    ap.add_argument("--exist_ok", action="store_true",
                    help="allow pushing into an existing repo (default: refuse)")
    ap.add_argument("--private", action="store_true",
                    help="create the repos as private")
    ap.add_argument("--dry_run", action="store_true",
                    help="validate everything, print the plan, do not upload")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not args.dry_run and not token:
        raise SystemExit("HF_TOKEN is not set. Export a write-scoped token "
                         "(e.g. from Colab Secrets) and re-run. It is never printed.")

    specs = [
        ModelSpec("grpo", args.grpo_ckpt, args.grpo_repo,
                  _HERE / "model_card_grpo.md"),
        ModelSpec("sft", args.sft_ckpt, args.sft_repo,
                  _HERE / "model_card_sft.md"),
    ]
    if args.only:
        specs = [s for s in specs if s.key == args.only]

    for spec in specs:
        upload(spec, user=args.user, token=token, exist_ok=args.exist_ok,
               private=args.private, dry_run=args.dry_run)

    if args.dry_run:
        print("\n[dry_run] all validations passed. Re-run without --dry_run "
              "(with HF_TOKEN set) to upload.")


if __name__ == "__main__":
    main()
