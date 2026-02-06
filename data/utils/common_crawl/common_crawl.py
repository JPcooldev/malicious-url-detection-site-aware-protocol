from __future__ import annotations
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
from urllib.parse import urlsplit, parse_qsl, urlencode, quote, urlparse
import re
import idna

import requests
from tqdm import tqdm

MAX_WORKERS = 3
CHUNK_SIZE = 1 << 20  # 1 MiB
TIMEOUT = 60  # seconds
PATTERN = re.compile(r"^CC-MAIN-(\d{4})-(\d{2})-(\d{5})\.gz$")

BASE_URL = "https://data.commoncrawl.org/cc-index/" "collections/{crawl}/indexes/cdx-{shard:05d}.gz"


def parse_shards(raw_list: List[str]) -> List[int]:
    """Convert CLI shard strings to validated integers 0-299."""
    ids: List[int] = []
    for s in raw_list:
        s = s.strip()
        if not s:
            continue
        try:
            num = int(s)  # accepts "43" or "00043"
        except ValueError:
            raise ValueError(f"Bad shard ID '{s}' (not an int)") from None
        ids.append(num)

    if not ids:
        raise RuntimeError("No shard IDs supplied.")
    return ids


def load_shard_paths(folder_path):
    files = []
    for fp in folder_path.iterdir():
        if not fp.is_file():
            continue
        m = PATTERN.match(fp.name)
        if not m:
            continue
        year, month, shard = m.group(1), m.group(2), m.group(3)
        files.append((fp, int(year), int(month), int(shard)))
    return files


def download_one(
    crawl_id: str,
    shard: int,
    out_path: Path,
    bar_position: int,
    total_jobs: int,
    job_idx: int,
) -> None:
    """
    Download one CDX shard to disk with its own tqdm progress bar.
    """
    url = BASE_URL.format(crawl=crawl_id, shard=shard)

    # HEAD first to size the progress bar
    with requests.head(url, timeout=TIMEOUT) as r:
        r.raise_for_status()
        length = int(r.headers.get("Content-Length", 0))

    desc = f"[{job_idx}/{total_jobs}] {out_path.name}"
    bar = tqdm(
        total=length,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        position=bar_position,
        desc=desc,
        leave=False,
    )

    try:
        with requests.get(url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))
    finally:
        bar.close()  # clears the line


def download_shards(
    crawl: str,
    shards: list[str],
    outdir: Path,
    workers: int = MAX_WORKERS,
) -> None:
    try:
        shard_ids = parse_shards(shards)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    total_jobs = len(shard_ids)
    futures = []
    next_bar_pos = 0

    outdir.mkdir(parents=True, exist_ok=True)
    shards_downloaded = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, shard in enumerate(shard_ids, 1):
            fname = f"{crawl}-{shard:05d}.gz"
            out = outdir / fname

            if out.exists():
                print(f"[{idx}/{total_jobs}] Skipping {out.name} (already exists).")
                continue
            shards_downloaded.append(shard)
            future = pool.submit(
                download_one,
                crawl,
                shard,
                out,
                next_bar_pos,
                total_jobs,
                idx,
            )
            futures.append(future)
            next_bar_pos = (next_bar_pos + 1) % workers

        for f in as_completed(futures):
            exc = f.exception()
            if exc:
                print(f"\nERROR: {exc}", file=sys.stderr)

    print("All requested downloads finished.")
    return shards_downloaded


def cdx_matches(metadata: dict) -> bool:
    """Return True if CDX JSON block meets the retention rules."""
    if metadata.get("status") != "200":
        return False
    url = metadata.get("url", "")
    if not url.startswith(("http://", "https://")):
        return False
    mime1 = metadata.get("mime", "").lower()
    mime2 = metadata.get("mime-detected", "").lower()
    return "text/html" in mime1 or "text/html" in mime2

def url_to_surt_hostname(url: str, collapse_www: bool = True) -> str | None:
    """
    Return SURT-style hostname (labels reversed, comma-separated), e.g.
      https://sub.example.co.uk  -> uk,co,example,sub
    Accepts scheme-less keys like 'example.com/path'.
    Returns None if no valid host is found.
    """
    if url is None:
        return None
    s = str(url).strip()
    if not s:
        return None

    # If it's a key like "example.com/..." add a dummy scheme so urlparse sees netloc
    if "://" not in s:
        if s.startswith("//"):
            s = "http:" + s
        else:
            s = "http://" + s

    p = urlparse(s)
    host = p.hostname or ""
    if not host:
        return None

    # IDNA normalize + lowercase
    try:
        host = host.encode("idna").decode("ascii").lower()
    except Exception:
        host = host.lower()

    if collapse_www and host.startswith("www."):
        host = host[4:]

    labels = host.split(".")
    labels.reverse()
    return ",".join(labels) if labels and labels[0] else None


# Tight list of ignorable tracking params
TRACKING_PARAMS = {
    "gclid", "gclsrc", "fbclid", "msclkid", "yclid", "twclid", "igshid",
    "mc_cid", "mc_eid", "mkt_tok",
    "pk_campaign", "pk_kwd", "pk_source",
}

# ;jsessionid in matrix params
_JSESSIONID_RE = re.compile(r";jsessionid=[A-Za-z0-9.\-]+", re.IGNORECASE)
# percent-encoding helpers
_PCT_ENC_RE = re.compile(r"%([0-9A-Fa-f]{2})")
_UNRESERVED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")

def _normalize_pct_encoding_path(path: str) -> str:
    """
    - Decode percent-encodings for UNRESERVED characters (%7E -> ~).
    - Uppercase hex for remaining encodings (%2f -> %2F).
    - Re-quote path without re-encoding valid percent triplets.
    """
    def _subst(m: re.Match) -> str:
        byte = int(m.group(1), 16)
        ch = chr(byte)
        if ch in _UNRESERVED:
            return ch                   # decode unreserved
        return f"%{m.group(1).upper()}" # keep encoded, uppercase hex

    path = _PCT_ENC_RE.sub(_subst, path)
    # IMPORTANT: keep '%' safe to avoid turning %2F into %252F
    return quote(path, safe="/-._~%")

def generate_url_matching_key(url: str) -> str:
    """
    Produce a stable, scheme-agnostic key for long-lived URL matching.

    Normalizations:
      - Strip default ports (:80 http, :443 https)
      - Ensure non-empty path (defaults to '/')
      - Remove fragment
      - Normalize percent-encoding for UNRESERVED chars; uppercase hex elsewhere
      - Remove empty query
      - Treat http/https as equivalent (omit scheme)
      - Collapse leading 'www.'
      - Sort query params by (key, value)
      - Hostname to IDNA/punycode, lowercase
      - Remove matrix ';jsessionid=...' from path
      - Drop tracking params: utm_*, gclid, gclsrc, fbclid, msclkid, yclid, twclid, igshid,
                              mc_cid, mc_eid, mkt_tok, pk_campaign, pk_kwd, pk_source
    """
    s = urlsplit(url)

    # ---- host/netloc ----
    host = (s.hostname or "").lower()
    if host:
        try:
            host = idna.encode(host).decode("ascii")
        except Exception:
            # If IDNA fails, keep the lowercase host as-is
            pass

    # strip leading www.
    if host.startswith("www."):
        host = host[4:]

    port = s.port
    if (s.scheme == "http" and port == 80) or (s.scheme == "https" and port == 443):
        port = None
    netloc = host if host else ""
    if port:
        netloc = f"{netloc}:{port}"

    # ---- path ----
    path = s.path or "/"
    # remove matrix ;jsessionid=...
    path = _JSESSIONID_RE.sub("", path)
    # percent-encoding normalization for path
    path = _normalize_pct_encoding_path(path)

    # ---- query ----
    kept = []
    if s.query:
        for k, v in parse_qsl(s.query, keep_blank_values=True):
            lk = k.lower()
            if lk.startswith("utm_") or lk in TRACKING_PARAMS:
                continue
            kept.append((k, v))
        kept.sort()  # by (key, value)

    query = urlencode(kept, doseq=True)

    # ---- build scheme-agnostic key (no fragment) ----
    # (do not include scheme to treat http/https as equivalent)
    if query:
        return f"{netloc}{path}?{query}"
    return f"{netloc}{path}"