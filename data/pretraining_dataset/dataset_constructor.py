import csv
import gzip
import json
import os
import sys
from pathlib import Path
from collections import Counter

from ..utils.common_crawl import PATTERN, cdx_matches
from ..utils.feature_extraction import get_sld


CHUNK_LINES = 100_000
MAX_PER_DOMAIN = 70
SECOND_MAX_PER_DOMAIN = 150
HARD_URL_MIN_LEN = 200

CSV_PATH = "pretraining_dataset.csv"
SHARDS_DIR = Path("cc_shards")
DOMAIN_COUNTS_JSON = "domain_counts.json"


def is_hard_url(url: str) -> bool:
    return len(url) > HARD_URL_MIN_LEN or "?" in url


def init_csv(path: str) -> tuple[set[tuple[int, int]], Counter]:
    """Initialise CSV and return (processed_months, domain_counts)."""
    processed: set[tuple[int, int]] = set()
    domain_counts: Counter = Counter()

    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["url", "year", "month"])
        return processed, domain_counts

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add((int(row["year"]), int(row["month"])))
            sld = get_sld(row["url"])
            if sld:
                domain_counts[sld] += 1
    return processed, domain_counts


def progress_msg(
    fname: str, file_idx: int, total_files: int, file_pct: float, kept: int, seen: int, overall_frac: float
):
    """Multi-line progress print."""
    sys.stdout.write(
        f"\rprocessing {fname}  ({file_idx}/{total_files} files)"
        f"\n   {file_pct:5.1f} % of file | {kept:,} kept / {seen:,} lines"
        f"\n   {overall_frac*100:5.1f} % of total workload        \n"
    )
    sys.stdout.flush()


def main() -> None:
    if not SHARDS_DIR.is_dir():
        print(f"No {SHARDS_DIR}/ directory found.")
        return

    processed_months, domain_counts = init_csv(CSV_PATH)

    # Discover candidate shard files
    files: list[tuple[Path, int, int]] = []
    for fn in SHARDS_DIR.iterdir():
        if not fn.is_file():
            continue
        m = PATTERN.match(fn.name)
        if not m:
            continue
        year, month = int(m.group(1)), int(m.group(3))
        if (year, month) in processed_months:
            continue
        files.append((fn, year, month))

    if not files:
        print("Nothing to do - all months already in pretraining_dataset.csv.")
        return

    files.sort(key=lambda x: (x[1], x[2]))  # chronological
    total_bytes = sum(f[0].stat().st_size for f in files)
    processed_bytes = 0
    hard_url_counter = 0
    with open(CSV_PATH, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        for idx, (fpath, year, month) in enumerate(files, start=1):
            fsize = fpath.stat().st_size
            kept = seen = 0

            with gzip.open(fpath, "rb") as g:
                while True:
                    raw = g.readline()
                    if not raw:
                        break

                    seen += 1
                    if seen % CHUNK_LINES == 0:
                        bytes_read = g.fileobj.tell()
                        overall_fraction = (processed_bytes + bytes_read) / total_bytes
                        progress_msg(
                            fpath.name, idx, len(files), bytes_read / fsize * 100, kept, seen, overall_fraction
                        )

                    try:
                        line = raw.decode("utf-8", "ignore")
                    except UnicodeDecodeError:
                        continue

                    jstart = line.find("{")
                    if jstart == -1:
                        continue
                    try:
                        meta = json.loads(line[jstart:])
                    except json.JSONDecodeError:
                        continue

                    if not cdx_matches(meta):
                        continue

                    url = meta["url"]
                    sld = get_sld(url)
                    if not sld:
                        continue

                    if domain_counts[sld] >= MAX_PER_DOMAIN:
                        if not is_hard_url(url):
                            continue
                        elif domain_counts[sld] >= SECOND_MAX_PER_DOMAIN:
                            continue
                        else:
                            hard_url_counter += 1

                    writer.writerow([url, year, month])
                    domain_counts[sld] += 1
                    kept += 1

            processed_bytes += fsize
            progress_msg(fpath.name, idx, len(files), 100.0, kept, seen, processed_bytes / total_bytes)
            processed_months.add((year, month))

    with open(DOMAIN_COUNTS_JSON, "w") as jf:
        json.dump(domain_counts, jf, indent=2, sort_keys=True)
    print("  Domain counts saved to", DOMAIN_COUNTS_JSON)
    print("hard urls", hard_url_counter)
    print(". Done. All eligible URLs appended to", CSV_PATH)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted - partial results kept.")
