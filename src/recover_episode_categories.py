"""Recover the episode→perturbation-category mapping from Sylvest/libero_plus_rlds.

`lerobot/libero_plus` (14,347 episodes) is a conversion of RLDS `libero_mix`
preserving episode order (prior investigation confirmed a 70/70 episode match
across 4 domains), and the RLDS-side `episode_metadata/file_path` retains the
perturbation category in the form

    .../pro_data/<category>/<suite>/<task>_demo.hdf5

The public LeRobot metadata drops this, so we read the split zip (~75 GB)
shard by shard over HTTP Range and recover each episode's (category, suite,
source, length, task) into a CSV.

Validation: assert that (task, length) matches lerobot/libero_plus's
meta/episodes for every episode. Even a single mismatch means the ordering
assumption is broken, so we fail.

Usage:
    python src/recover_episode_categories.py --output_dir <dir> [--workers 8]

Each shard is cached to <output_dir>/shards/*.json, so a rerun resumes after
an interruption.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import struct
import time
import zlib
from pathlib import Path

import requests

RLDS_REPO = "Sylvest/libero_plus_rlds"
RLDS_PARTS = [  # split zip: concatenating .z01, .z02, .zip in this order yields one zip
    "libero_plus_mixdata.z01",
    "libero_plus_mixdata.z02",
    "libero_plus_mixdata.zip",
]
TFDS_PREFIX = "libero_mix/1.0.0"
N_SHARDS = 1024

# RLDS pro_data/<dir> → this repo's canonical category names (matches make_heldout_split.py)
CATEGORY_MAP = {
    "camera_view": "camera_pose",
    "light": "lighting",
    "noise": "sensor_noise",
    "language": "language",
    "env": "env",
}


# ------------------------------------------------------------
# HTTP split-zip reader
# ------------------------------------------------------------
class RemoteSplitZip:
    """Reads a split zip over HTTP Range. Downloads are per shard (~60-70 MB)."""

    def __init__(self, urls: list[str], session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.urls = urls
        self.sizes = [self._content_length(u) for u in urls]
        self.starts = [0]
        for s in self.sizes[:-1]:
            self.starts.append(self.starts[-1] + s)
        self.total = sum(self.sizes)
        self.entries = self._read_central_directory()

    def _content_length(self, url: str) -> int:
        for attempt in range(5):
            try:
                r = self.session.head(url, allow_redirects=True, timeout=30)
                r.raise_for_status()
                return int(r.headers["Content-Length"])
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("unreachable")

    def _fetch_part(self, part: int, offset: int, n: int) -> bytes:
        headers = {"Range": f"bytes={offset}-{offset + n - 1}"}
        for attempt in range(5):
            try:
                r = self.session.get(
                    self.urls[part], headers=headers, allow_redirects=True, timeout=120
                )
                r.raise_for_status()
                data = r.content
                if len(data) != n:
                    raise IOError(f"short read: got {len(data)}, want {n}")
                return data
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("unreachable")

    def read_abs(self, abs_offset: int, n: int) -> bytes:
        """Absolute-offset read spanning part boundaries."""
        out = b""
        pos = abs_offset
        remaining = n
        while remaining > 0:
            part = max(i for i, s in enumerate(self.starts) if s <= pos)
            local = pos - self.starts[part]
            take = min(remaining, self.sizes[part] - local)
            out += self._fetch_part(part, local, take)
            pos += take
            remaining -= take
        return out

    # ---- central directory ----
    def _read_central_directory(self) -> dict[str, dict]:
        tail_len = min(1 << 20, self.sizes[-1])
        tail = self._fetch_part(len(self.urls) - 1, self.sizes[-1] - tail_len, tail_len)

        eocd_pos = tail.rfind(b"PK\x05\x06")
        if eocd_pos < 0:
            raise RuntimeError("EOCD not found")
        cd_size, cd_offset = struct.unpack_from("<II", tail, eocd_pos + 12)
        cd_disk = struct.unpack_from("<H", tail, eocd_pos + 6)[0]
        n_entries = struct.unpack_from("<H", tail, eocd_pos + 10)[0]

        # zip64 (required: some parts exceed 32 GB)
        loc_pos = tail.rfind(b"PK\x06\x07", 0, eocd_pos)
        if loc_pos >= 0:
            eocd64_disk, eocd64_off = struct.unpack_from("<IQ", tail, loc_pos + 4)
            eocd64 = self.read_abs(self.starts[eocd64_disk] + eocd64_off, 56)
            assert eocd64[:4] == b"PK\x06\x06", "bad EOCD64"
            n_entries = struct.unpack_from("<Q", eocd64, 32)[0]
            cd_size = struct.unpack_from("<Q", eocd64, 40)[0]
            cd_disk_off = struct.unpack_from("<Q", eocd64, 48)[0]
            cd_disk = struct.unpack_from("<I", eocd64, 20)[0]
            cd_offset = cd_disk_off

        cd = self.read_abs(self.starts[cd_disk] + cd_offset, cd_size)

        entries: dict[str, dict] = {}
        p = 0
        for _ in range(n_entries):
            assert cd[p : p + 4] == b"PK\x01\x02", f"bad CD entry at {p}"
            method = struct.unpack_from("<H", cd, p + 10)[0]
            csize, usize = struct.unpack_from("<II", cd, p + 20)
            name_len, extra_len, comment_len = struct.unpack_from("<HHH", cd, p + 28)
            disk = struct.unpack_from("<H", cd, p + 34)[0]
            lho = struct.unpack_from("<I", cd, p + 42)[0]
            name = cd[p + 46 : p + 46 + name_len].decode("utf-8")
            extra = cd[p + 46 + name_len : p + 46 + name_len + extra_len]

            # zip64 extra field: only the fields that are 0xFFFF/0xFFFFFFFF appear, in this order
            q = 0
            while q + 4 <= len(extra):
                tag, size = struct.unpack_from("<HH", extra, q)
                if tag == 0x0001:
                    body = extra[q + 4 : q + 4 + size]
                    r = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack_from("<Q", body, r)[0]
                        r += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack_from("<Q", body, r)[0]
                        r += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack_from("<Q", body, r)[0]
                        r += 8
                    if disk == 0xFFFF:
                        disk = struct.unpack_from("<I", body, r)[0]
                        r += 4
                q += 4 + size

            entries[name] = {
                "method": method,
                "csize": csize,
                "usize": usize,
                "disk": disk,
                "lho": lho,
            }
            p += 46 + name_len + extra_len + comment_len
        return entries

    def resolve(self, suffix: str) -> str:
        """Full paths in the zip are absolute paths from the build cluster, so look up by suffix."""
        if suffix in self.entries:
            return suffix
        matches = [n for n in self.entries if n.endswith("/" + suffix)]
        if len(matches) != 1:
            raise KeyError(f"{suffix}: {len(matches)} matches in zip")
        return matches[0]

    def open_entry(self, name: str) -> bytes:
        e = self.entries[self.resolve(name)]
        header_abs = self.starts[e["disk"]] + e["lho"]
        lfh = self.read_abs(header_abs, 30)
        assert lfh[:4] == b"PK\x03\x04", f"bad local header for {name}"
        name_len, extra_len = struct.unpack_from("<HH", lfh, 26)
        data = self.read_abs(header_abs + 30 + name_len + extra_len, e["csize"])
        if e["method"] == 8:
            return zlib.decompress(data, -15)
        if e["method"] == 0:
            return data
        raise ValueError(f"unsupported compression method {e['method']}")


# ------------------------------------------------------------
# Minimal TFRecord + tf.Example parser (skips image byte strings)
# ------------------------------------------------------------
def iter_tfrecords(data: bytes):
    pos = 0
    n = len(data)
    while pos < n:
        (length,) = struct.unpack_from("<Q", data, pos)
        pos += 12  # 8 length + 4 masked crc
        yield memoryview(data)[pos : pos + length]
        pos += length + 4  # payload + 4 data crc


def _read_varint(buf, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _skip_field(buf, pos: int, wire_type: int) -> int:
    if wire_type == 0:
        _, pos = _read_varint(buf, pos)
        return pos
    if wire_type == 1:
        return pos + 8
    if wire_type == 2:
        ln, pos = _read_varint(buf, pos)
        return pos + ln
    if wire_type == 5:
        return pos + 4
    raise ValueError(f"unexpected wire type {wire_type}")


def _parse_feature(buf, start: int, end: int, want_values: bool):
    """Read Feature { BytesList=1 / FloatList=2 / Int64List=3 }.
    Returns: (kind, values or count)."""
    pos = start
    while pos < end:
        tag, pos = _read_varint(buf, pos)
        field, wt = tag >> 3, tag & 7
        if wt != 2:
            pos = _skip_field(buf, pos, wt)
            continue
        ln, pos = _read_varint(buf, pos)
        body_end = pos + ln
        if field == 1:  # BytesList { repeated bytes value = 1 }
            values = []
            count = 0
            q = pos
            while q < body_end:
                t, q = _read_varint(buf, q)
                if t & 7 != 2:
                    q = _skip_field(buf, q, t & 7)
                    continue
                vlen, q = _read_varint(buf, q)
                if want_values:
                    values.append(bytes(buf[q : q + vlen]))
                count += 1
                q += vlen
            return "bytes", values if want_values else count
        if field == 3:  # Int64List { repeated int64 value = 1 (packed or not) }
            count = 0
            q = pos
            while q < body_end:
                t, q = _read_varint(buf, q)
                if t & 7 == 2:  # packed
                    vlen, q = _read_varint(buf, q)
                    stop = q + vlen
                    while q < stop:
                        _, q = _read_varint(buf, q)
                        count += 1
                else:
                    _, q = _read_varint(buf, q)
                    count += 1
            return "int64", count
        pos = body_end
    return None, None


def parse_episode(record) -> dict:
    """Extract file_path / language / T from a tf.Example. Everything else (images etc.) is skipped."""
    buf = record
    pos = 0
    end = len(buf)
    out: dict = {}
    # Example { Features features = 1 } → Features { map<string, Feature> feature = 1 }
    tag, pos = _read_varint(buf, pos)
    assert tag >> 3 == 1 and tag & 7 == 2, "not a tf.Example"
    _, pos = _read_varint(buf, pos)
    while pos < end:
        tag, pos = _read_varint(buf, pos)
        if tag >> 3 != 1 or tag & 7 != 2:
            pos = _skip_field(buf, pos, tag & 7)
            continue
        entry_len, pos = _read_varint(buf, pos)
        entry_end = pos + entry_len
        key = None
        q = pos
        while q < entry_end:
            t, q = _read_varint(buf, q)
            f, wt = t >> 3, t & 7
            if wt != 2:
                q = _skip_field(buf, q, wt)
                continue
            ln, q = _read_varint(buf, q)
            if f == 1:
                key = bytes(buf[q : q + ln]).decode("utf-8")
                q += ln
            elif f == 2 and key is not None:
                if key == "episode_metadata/file_path":
                    _, vals = _parse_feature(buf, q, q + ln, want_values=True)
                    out["file_path"] = vals[0].decode("utf-8")
                elif key == "steps/language_instruction":
                    _, vals = _parse_feature(buf, q, q + ln, want_values=True)
                    out["task"] = vals[0].decode("utf-8")
                elif key == "steps/is_last":
                    _, count = _parse_feature(buf, q, q + ln, want_values=False)
                    out["length"] = count
                q += ln
            else:
                q += ln
        pos = entry_end
    return out


def categorize(file_path: str) -> tuple[str, str, str]:
    """file_path → (category, suite, source_file)."""
    parts = file_path.split("/")
    i = parts.index("pro_data")
    raw_cat, suite, fname = parts[i + 1], parts[i + 2], parts[i + 3]
    return CATEGORY_MAP.get(raw_cat, raw_cat), suite, fname


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def shard_name(i: int) -> str:
    return f"{TFDS_PREFIX}/libero_mix-train.tfrecord-{i:05d}-of-{N_SHARDS:05d}"


def process_shard(rz: RemoteSplitZip, i: int, cache_dir: Path) -> list[dict]:
    cache = cache_dir / f"shard_{i:05d}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    data = rz.open_entry(shard_name(i))
    episodes = []
    for rec in iter_tfrecords(data):
        ep = parse_episode(rec)
        cat, suite, src = categorize(ep["file_path"])
        episodes.append(
            {
                "category": cat,
                "suite": suite,
                "source_file": src,
                "length": ep["length"],
                "task": ep["task"],
            }
        )
    cache.write_text(json.dumps(episodes))
    return episodes


def load_lerobot_meta(cache_dir: Path) -> list[dict]:
    """Read lerobot/libero_plus meta/episodes for (task, length) cross-checking."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    repo_dir = snapshot_download(
        repo_id="lerobot/libero_plus",
        repo_type="dataset",
        allow_patterns=["meta/*"],
        cache_dir=str(cache_dir),
    )
    meta_dir = Path(repo_dir) / "meta"
    rows: list[dict] = []
    for path in sorted((meta_dir / "episodes").rglob("*.parquet")):
        rows.extend(pq.read_table(path, columns=["episode_index", "tasks", "length"]).to_pylist())
    rows.sort(key=lambda r: r["episode_index"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache_dir", type=Path, default=Path("/content/hf_cache"))
    ap.add_argument("--skip_validation", action="store_true", help="skip cross-checking against the lerobot meta")
    args = ap.parse_args()

    shard_cache = args.output_dir / "shards"
    shard_cache.mkdir(parents=True, exist_ok=True)

    urls = [
        f"https://huggingface.co/datasets/{RLDS_REPO}/resolve/main/{p}" for p in RLDS_PARTS
    ]
    rz = RemoteSplitZip(urls)
    print(f"[recover] zip total {rz.total / 1e9:.1f} GB, {len(rz.entries)} entries")

    info = json.loads(rz.open_entry(f"{TFDS_PREFIX}/dataset_info.json").decode("utf-8"))
    shard_lengths = [int(x) for x in info["splits"][0]["shardLengths"]]
    n_total = sum(shard_lengths)
    print(f"[recover] {len(shard_lengths)} shards, {n_total} episodes")

    t0 = time.time()
    results: dict[int, list[dict]] = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_shard, rz, i, shard_cache): i for i in range(len(shard_lengths))}
        done = 0
        for fut in cf.as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            if len(results[i]) != shard_lengths[i]:
                raise RuntimeError(
                    f"shard {i}: parsed {len(results[i])} episodes, expected {shard_lengths[i]}"
                )
            done += 1
            if done % 32 == 0 or done == len(shard_lengths):
                rate = done / (time.time() - t0)
                eta = (len(shard_lengths) - done) / max(rate, 1e-9)
                print(f"[recover] {done}/{len(shard_lengths)} shards ({eta / 60:.0f} min left)")

    rows: list[dict] = []
    for i in range(len(shard_lengths)):
        for ep in results[i]:
            rows.append({"episode_index": len(rows), **ep})
    assert len(rows) == n_total

    # ---- Validation: do (task, length) match the lerobot/libero_plus meta? ----
    if not args.skip_validation:
        meta = load_lerobot_meta(args.cache_dir)
        assert len(meta) == len(rows), f"episode count mismatch: {len(meta)} vs {len(rows)}"
        mismatches = []
        for m, r in zip(meta, rows):
            task = m["tasks"][0] if isinstance(m["tasks"], list) else m["tasks"]
            if task != r["task"] or int(m["length"]) != int(r["length"]):
                mismatches.append((m["episode_index"], task, m["length"], r["task"], r["length"]))
        if mismatches:
            raise RuntimeError(
                f"{len(mismatches)} episodes mismatch (ordering assumption broken); "
                f"first: {mismatches[0]}"
            )
        print(f"[recover] validation OK: all {len(rows)} (task, length) pairs match lerobot meta")

    out_csv = args.output_dir / "episode_categories.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["episode_index", "category", "suite", "source_file", "length", "task"]
        )
        w.writeheader()
        w.writerows(rows)

    summary: dict[str, int] = {}
    for r in rows:
        summary[r["category"]] = summary.get(r["category"], 0) + 1
    (args.output_dir / "categories_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[recover] wrote {out_csv}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
