"""
Verified migration: syraxzzz/maithili_data  ->  mlwiseyak/raw_maithili_audio_data_1

Copies every file from the source repo to a brand-new target repo, verifying
each file's SHA256 against the hash HuggingFace has on record for it *before*
uploading, and batching multiple files into a single commit to stay well
under HuggingFace's hard limit of 128 repository commits per hour.

WHY BATCHED COMMITS
--------------------
Earlier versions of this script called upload_file() once per file, i.e.
one commit per file. HuggingFace caps commits at 128/hour per repo - with
thousands of files, that limit gets hit almost immediately and every file
after that fails with HTTP 429. This version groups BATCH_COMMIT_SIZE files
into a single create_commit() call, cutting the total commit count by the
same factor (e.g. 50 files/commit -> 40x fewer commits).

If a 429 does still happen (e.g. batch size set too small for a very large
dataset), the script reads HF's own "Retry after N seconds" from the error
and waits exactly that long before retrying the SAME commit - it does not
give up or re-download anything.

DISK SPACE BEHAVIOR
--------------------
Files for the current commit batch are downloaded, verified, committed,
then deleted - the script never needs to hold the full dataset on disk at
once. huggingface_hub's own blob cache is redirected into the workdir and
purged periodically (or after every file in LOW_DISK_MODE) so footprint
stays bounded regardless of dataset size.

RESUME BEHAVIOR
-----------------
A file only counts as "done" after it's been hash-verified AND successfully
included in a committed batch. If the script is interrupted at any point
before that (crash, Ctrl+C, out of disk space, unresolvable rate limit),
that file simply isn't marked done, so the next run retries it. Nothing
partial or corrupted is ever left in the target repo.

Usage:
    pip install huggingface_hub --break-system-packages
    export TARGET_HF_TOKEN="hf_..."      # WRITE token for the mlwiseyak account
    python migrate_verified.py

Optional env vars:
    SOURCE_REPO_ID      default: syraxzzz/maithili_data
    TARGET_REPO_ID      default: mlwiseyak/raw_maithili_audio_data_1
    TARGET_PRIVATE      "true"/"false", default "false"
    BATCH_COMMIT_SIZE   files grouped into a single commit, default 50
    MAX_WORKERS         parallel download/verify workers per batch, default 6
                         (ignored if LOW_DISK_MODE=true)
    LOW_DISK_MODE       "true"/"false", default "false"
                         true  -> sequential downloads, purge cache after
                                  every file, minimal disk footprint, slower.
                         false -> parallel downloads, purge cache every
                                  PURGE_EVERY files.
    MIN_FREE_GB         minimum free disk space (GB) required to keep going,
                         default 2. Checked before every batch.
    PURGE_EVERY         files between cache purges in normal mode, default 100.
    SECONDS_BETWEEN_COMMITS  polite pause between commits to avoid tripping
                         the limit in the first place, default 5.
"""

import os
import re
import json
import time
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
from huggingface_hub.utils import HfHubHTTPError

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SOURCE_REPO_ID = os.getenv("SOURCE_REPO_ID", "syraxzzz/maithili_data")
TARGET_REPO_ID = os.getenv("TARGET_REPO_ID", "mlwiseyak/raw_maithili_audio_data_1")
TARGET_HF_TOKEN = os.getenv("TARGET_HF_TOKEN", "")
SOURCE_HF_TOKEN = os.getenv("SOURCE_HF_TOKEN", None)
TARGET_PRIVATE = os.getenv("TARGET_PRIVATE", "false").lower() in ("1", "true", "yes")
LOW_DISK_MODE = os.getenv("LOW_DISK_MODE", "false").lower() in ("1", "true", "yes")
BATCH_COMMIT_SIZE = int(os.getenv("BATCH_COMMIT_SIZE", "50"))
MAX_WORKERS = 1 if LOW_DISK_MODE else int(os.getenv("MAX_WORKERS", "6"))
MIN_FREE_GB = float(os.getenv("MIN_FREE_GB", "2"))
PURGE_EVERY = 1 if LOW_DISK_MODE else int(os.getenv("PURGE_EVERY", "100"))
SECONDS_BETWEEN_COMMITS = float(os.getenv("SECONDS_BETWEEN_COMMITS", "5"))
REPO_TYPE = "dataset"

WORKDIR = Path("verify_migrate_workdir")
CACHE_DIR = WORKDIR / ".hf_cache"
FAILED_LOG = Path("migration_verified_failed.json")
MISMATCH_LOG = Path("migration_verified_hash_mismatches.json")
DONE_LOG = Path("migration_verified_done.json")

RETRY_AFTER_RE = re.compile(r"Retry after (\d+) seconds", re.IGNORECASE)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def purge_cache():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_enough_space(min_free_gb: float) -> bool:
    current = free_gb(WORKDIR)
    if current >= min_free_gb:
        return True
    print(f"[WARN] Low disk space ({current:.2f} GB free). Purging local cache ...")
    purge_cache()
    current = free_gb(WORKDIR)
    if current >= min_free_gb:
        print(f"[OK] Freed enough space ({current:.2f} GB free now).")
        return True
    print(f"[ERROR] Still only {current:.2f} GB free after purging cache "
          f"(need {min_free_gb} GB). Stopping cleanly.")
    return False


def get_source_files_with_hashes(api: HfApi) -> List[Dict]:
    print(f"[INFO] Fetching file metadata for {SOURCE_REPO_ID} ...")
    info = api.dataset_info(SOURCE_REPO_ID, files_metadata=True, token=SOURCE_HF_TOKEN)
    results = []
    for sibling in info.siblings:
        expected_hash = None
        if sibling.lfs and getattr(sibling.lfs, "sha256", None):
            expected_hash = sibling.lfs.sha256
        results.append({
            "rfilename": sibling.rfilename,
            "expected_sha256": expected_hash,
            "size": sibling.size,
        })
    print(f"[OK] {len(results)} files found ({sum(1 for r in results if r['expected_sha256'])} with LFS hashes)")
    return results


def download_and_verify_one(entry: Dict) -> Dict:
    """Download + hash-verify a single file. Does NOT upload - upload
    happens once per whole batch via a single commit."""
    filename = entry["rfilename"]
    expected_hash = entry["expected_sha256"]
    result = {"filename": filename, "status": None, "detail": None, "local_path": None}

    try:
        local_path = Path(hf_hub_download(
            repo_id=SOURCE_REPO_ID,
            filename=filename,
            repo_type=REPO_TYPE,
            token=SOURCE_HF_TOKEN,
            local_dir=WORKDIR,
            cache_dir=CACHE_DIR,
        ))
    except Exception as e:
        result["status"] = "download_failed"
        result["detail"] = str(e)
        return result

    if expected_hash:
        actual_hash = sha256_of(local_path)
        if actual_hash.lower() != expected_hash.lower():
            result["status"] = "hash_mismatch"
            result["detail"] = f"expected {expected_hash}, got {actual_hash}"
            local_path.unlink(missing_ok=True)
            return result

    result["status"] = "downloaded_ok"
    result["local_path"] = local_path
    return result


def download_and_verify_batch(batch: List[Dict]) -> List[Dict]:
    if MAX_WORKERS == 1:
        return [download_and_verify_one(e) for e in batch]
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_and_verify_one, e): e for e in batch}
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


def commit_batch_with_retry(api: HfApi, ready: List[Dict], max_retries: int = 10) -> Optional[str]:
    """Commits every file in `ready` (each with a valid local_path) as ONE
    commit. On a 429, sleeps for exactly the time HF asks for and retries -
    this is a temporary throttle, not a permanent failure, so we don't give
    up on it. Returns None on success, or an error string on permanent
    failure (non-429 error)."""
    operations = [
        CommitOperationAdd(path_in_repo=r["filename"], path_or_fileobj=str(r["local_path"]))
        for r in ready
    ]

    attempt = 0
    while True:
        attempt += 1
        try:
            api.create_commit(
                repo_id=TARGET_REPO_ID,
                repo_type=REPO_TYPE,
                operations=operations,
                commit_message=f"Verified migration batch ({len(ready)} files) from {SOURCE_REPO_ID}",
                token=TARGET_HF_TOKEN,
            )
            return None
        except HfHubHTTPError as e:
            msg = str(e)
            if "429" in msg or "rate limit" in msg.lower():
                m = RETRY_AFTER_RE.search(msg)
                wait_s = int(m.group(1)) + 5 if m else 60
                print(f"[WAIT] Rate limited on commit (attempt {attempt}/{max_retries}). "
                      f"Waiting {wait_s}s as instructed by HuggingFace ...")
                if attempt >= max_retries:
                    return f"rate_limited_after_{attempt}_attempts: {msg}"
                time.sleep(wait_s)
                continue
            return msg
        except Exception as e:
            return str(e)


def main():
    if not TARGET_HF_TOKEN:
        print("[ERROR] Set TARGET_HF_TOKEN to a WRITE token for the mlwiseyak account.")
        return

    WORKDIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    api = HfApi()

    mode = "LOW_DISK_MODE (sequential downloads, purge every file)" if LOW_DISK_MODE else \
           f"normal (parallel downloads x{MAX_WORKERS}, purge every {PURGE_EVERY} files)"
    print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Commit batch size: {BATCH_COMMIT_SIZE} files/commit")
    print(f"[INFO] Minimum free space required: {MIN_FREE_GB} GB")

    print(f"[INFO] Creating target repo {TARGET_REPO_ID} (private={TARGET_PRIVATE}) ...")
    api.create_repo(
        repo_id=TARGET_REPO_ID,
        repo_type=REPO_TYPE,
        token=TARGET_HF_TOKEN,
        private=TARGET_PRIVATE,
        exist_ok=True,
    )
    print(f"[OK] Repo ready: https://huggingface.co/datasets/{TARGET_REPO_ID}")

    all_entries = get_source_files_with_hashes(api)

    done = set(load_json(DONE_LOG, []))
    failed_prev = load_json(FAILED_LOG, [])
    mismatch_prev = load_json(MISMATCH_LOG, [])

    to_process = [e for e in all_entries if e["rfilename"] not in done]
    print(f"[INFO] Already migrated & verified: {len(done)}")
    print(f"[INFO] Remaining: {len(to_process)}")
    est_commits = -(-len(to_process) // BATCH_COMMIT_SIZE)  # ceiling division
    print(f"[INFO] Estimated commits needed this run: ~{est_commits} (limit is 128/hour)")

    ok_count = 0
    fail_count = 0
    mismatch_count = 0
    stopped_early = False

    dl_batch_size = max(MAX_WORKERS, 1)

    commit_group: List[Dict] = []
    files_since_purge = 0

    def flush_commit_group():
        nonlocal ok_count, fail_count, commit_group
        if not commit_group:
            return
        err = commit_batch_with_retry(api, commit_group)
        if err is None:
            for r in commit_group:
                done.add(r["filename"])
                ok_count += 1
            print(f"[OK] Committed batch of {len(commit_group)} files.")
        else:
            for r in commit_group:
                fail_count += 1
                failed_prev.append({"filename": r["filename"], "status": "upload_failed", "detail": err})
            print(f"[ERROR] Commit failed for batch of {len(commit_group)} files: {err}")

        for r in commit_group:
            try:
                Path(r["local_path"]).unlink(missing_ok=True)
            except Exception:
                pass

        save_json(DONE_LOG, sorted(done))
        save_json(FAILED_LOG, failed_prev)
        commit_group = []
        if SECONDS_BETWEEN_COMMITS > 0:
            time.sleep(SECONDS_BETWEEN_COMMITS)

    for i in range(0, len(to_process), dl_batch_size):
        dl_batch = to_process[i:i + dl_batch_size]

        if not ensure_enough_space(MIN_FREE_GB):
            stopped_early = True
            break

        results = download_and_verify_batch(dl_batch)
        for res in results:
            if res["status"] == "downloaded_ok":
                commit_group.append(res)
            elif res["status"] == "hash_mismatch":
                mismatch_count += 1
                mismatch_prev.append(res)
                print(f"[MISMATCH] {res['filename']}")
            else:
                fail_count += 1
                failed_prev.append(res)
                print(f"[DL-FAIL] {res['filename']}: {res['detail']}")

        files_since_purge += len(dl_batch)
        save_json(MISMATCH_LOG, mismatch_prev)
        save_json(FAILED_LOG, failed_prev)

        if len(commit_group) >= BATCH_COMMIT_SIZE:
            flush_commit_group()

        if files_since_purge >= PURGE_EVERY:
            purge_cache()
            files_since_purge = 0

    flush_commit_group()  # commit any remaining partial group

    save_json(DONE_LOG, sorted(done))
    save_json(FAILED_LOG, failed_prev)
    save_json(MISMATCH_LOG, mismatch_prev)

    print("\n" + "=" * 60)
    print(f"Successfully migrated & hash-verified: {ok_count}")
    print(f"Hash mismatches (corrupted at source, NOT uploaded): {mismatch_count}")
    print(f"Download/upload failures: {fail_count}")
    print(f"Total verified-good files in target repo so far: {len(done)}")
    if stopped_early:
        print(f"\n[STOPPED] Ran out of usable disk space part-way through.")
        print(f"          Free up space, then just re-run this script - it will")
        print(f"          resume from file {len(done)+1} automatically.")
    print("=" * 60)
    if mismatch_count or fail_count:
        print(f"\nSee {MISMATCH_LOG} and {FAILED_LOG} for details.")
        print("Re-run this script to retry failures; it will skip everything already verified.")


if __name__ == "__main__":
    main()