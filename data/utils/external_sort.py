import csv
import heapq
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterable, List, Optional, Sequence, Union

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback progress bar for when tqdm is not available
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc or ""
            self.n = 0
            self._printed_progress = set()
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            if self.total:
                print(f"\r{self.desc}: 100%", flush=True)
        
        def __iter__(self):
            if self.iterable:
                for item in self.iterable:
                    yield item
                    self.update(1)
            return self
        
        def update(self, n=1):
            self.n += n
            if self.total:
                pct = int(100 * self.n / self.total)
                if pct % 10 == 0 and pct not in self._printed_progress:
                    print(f"\r{self.desc}: {pct}%", end="", flush=True)
                    self._printed_progress.add(pct)


def external_csv_sort(
    input_csv: Union[str, os.PathLike],
    output_csv: Union[str, os.PathLike],
    key_cols: Union[str, int, Sequence[Union[str, int]]],
    *,
    has_header: bool = True,
    delimiter: str = ",",
    quotechar: str = '"',
    chunk_rows: int = 500_000,
    key_fn: Optional[Callable[[List[str]], Union[str, float, int, tuple]]] = None,
    reverse: bool = False,
    tempdir: Optional[Union[str, os.PathLike]] = None,
    encoding: str = "utf-8",
    newline: str = "",
    show_progress: bool = True
) -> None:
    """
    External sort of a CSV by one or more columns with progress tracking.
    - Temp chunk files are created in a system temp directory (or `tempdir` if provided), not CWD.
    - Memory use is bounded by `chunk_rows`.

    Args:
        input_csv: Path to input CSV.
        output_csv: Path to output CSV.
        key_cols: Column name, index, or sequence of them to sort by. If `key_fn` is provided,
                  `key_cols` is ignored.
        has_header: Whether the CSV has a header row.
        delimiter, quotechar: CSV dialect controls.
        chunk_rows: Approx rows per in-memory chunk before spill to disk.
        key_fn: Custom key extractor: receives a full row (list[str]) and returns a sortable key.
        reverse: Sort descending if True.
        tempdir: Optional directory to place temporary files.
        encoding, newline: Passed to file open calls.
        show_progress: Whether to show progress bars (requires tqdm).
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Estimate total rows for progress tracking
    def _count_csv_rows(filepath):
        if not show_progress:
            return None
        try:
            with filepath.open("r", encoding=encoding, newline=newline) as f:
                reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
                count = sum(1 for _ in reader)
                return count - (1 if has_header else 0)  # subtract header if present
        except:
            return None

    total_rows = _count_csv_rows(input_csv) if show_progress else None

    class _RevKey:
        __slots__ = ("v",)
        def __init__(self, v): self.v = v
        def __lt__(self, other): return self.v > other.v  # reverse order in a min-heap

    def _make_key_selector(header: Optional[List[str]]):
        if key_fn is not None:
            return (lambda row: _RevKey(key_fn(row)) if reverse else key_fn(row))

        # normalize key_cols to indices
        if isinstance(key_cols, (str, int)):
            cols = [key_cols]
        else:
            cols = list(key_cols)

        if header is not None:
            name_to_idx = {name: i for i, name in enumerate(header)}
            idxs = [ (name_to_idx[c] if isinstance(c, str) else int(c)) for c in cols ]
        else:
            idxs = [ (int(c) if isinstance(c, int) or str(c).isdigit() else None) for c in cols ]
            if any(i is None for i in idxs):
                raise ValueError("Without a header, key_cols must be integer indices.")

        def _k(row):
            k = tuple(row[i] for i in idxs) if len(idxs) > 1 else row[idxs[0]]
            return _RevKey(k) if reverse else k

        return _k

    def _read_chunk(reader, n, pbar=None):
        chunk = []
        for _ in range(n):
            try:
                row = next(reader)
            except StopIteration:
                break
            chunk.append(row)
            if pbar:
                pbar.update(1)
        return chunk

    def _write_rows(fp, rows):
        w = csv.writer(fp, delimiter=delimiter, quotechar=quotechar, lineterminator="\n")
        for r in rows:
            w.writerow(r)

    with TemporaryDirectory(dir=tempdir) as tdir:
        tdir = Path(tdir)
        chunk_paths: List[Path] = []
        header: Optional[List[str]] = None

        # Phase 1: Read and sort chunks
        with input_csv.open("r", encoding=encoding, newline=newline) as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            
            if has_header:
                try:
                    header = next(reader)
                except StopIteration:
                    # empty file
                    with output_csv.open("w", encoding=encoding, newline="\n") as out_f:
                        pass
                    return

            key_selector = _make_key_selector(header)

            # Progress bar for reading and sorting chunks
            read_pbar = None
            if show_progress and total_rows:
                read_pbar = tqdm(
                    total=total_rows, 
                    desc="Reading & sorting chunks", 
                    unit="rows"
                )

            try:
                while True:
                    chunk = _read_chunk(reader, chunk_rows, read_pbar)
                    if not chunk:
                        break
                    
                    # Sort chunk in memory
                    chunk.sort(key=key_selector)

                    # Write sorted chunk to temp file
                    cpath = tdir / f"chunk_{len(chunk_paths):06d}.csv"
                    with cpath.open("w", encoding=encoding, newline="\n") as cf:
                        _write_rows(cf, chunk)
                    chunk_paths.append(cpath)

            finally:
                if read_pbar:
                    read_pbar.close()

        # If only one chunk or no chunks, handle directly
        if len(chunk_paths) == 0:
            # nothing to merge but may have header-only
            with output_csv.open("w", encoding=encoding, newline="\n") as out_f:
                w = csv.writer(out_f, delimiter=delimiter, quotechar=quotechar, lineterminator="\n")
                if has_header and header is not None:
                    w.writerow(header)
            if show_progress:
                print("✓ Empty file processed")
            return
        elif len(chunk_paths) == 1:
            with output_csv.open("w", encoding=encoding, newline="\n") as out_f:
                w = csv.writer(out_f, delimiter=delimiter, quotechar=quotechar, lineterminator="\n")
                if has_header and header is not None:
                    w.writerow(header)
                with chunk_paths[0].open("r", encoding=encoding, newline=newline) as cf:
                    for row in csv.reader(cf, delimiter=delimiter, quotechar=quotechar):
                        w.writerow(row)
            if show_progress:
                print("✓ Single chunk processed (no merge needed)")
            return

        # Phase 2: k-way merge
        files = [p.open("r", encoding=encoding, newline=newline) for p in chunk_paths]
        readers = [csv.reader(fp, delimiter=delimiter, quotechar=quotechar) for fp in files]

        # prime heap
        heap = []
        for i, rdr in enumerate(readers):
            try:
                row = next(rdr)
            except StopIteration:
                continue
            k = key_selector(row)
            # Use a monotonic tiebreaker to avoid comparing rows when keys equal
            heap.append((k, i, row))
        heapq.heapify(heap)

        # Progress bar for merging
        merge_pbar = None
        if show_progress and total_rows:
            merge_pbar = tqdm(
                total=total_rows,
                desc="Merging sorted chunks",
                unit="rows"
            )

        try:
            with output_csv.open("w", encoding=encoding, newline="\n") as out_f:
                w = csv.writer(out_f, delimiter=delimiter, quotechar=quotechar, lineterminator="\n")
                if has_header and header is not None:
                    w.writerow(header)

                while heap:
                    _, src_i, row = heapq.heappop(heap)
                    w.writerow(row)
                    if merge_pbar:
                        merge_pbar.update(1)
                    
                    try:
                        nxt = next(readers[src_i])
                    except StopIteration:
                        continue
                    k = key_selector(nxt)
                    heapq.heappush(heap, (k, src_i, nxt))

        finally:
            if merge_pbar:
                merge_pbar.close()
            for fp in files:
                fp.close()

        if show_progress:
            print(f"✓ Sort complete: {len(chunk_paths)} chunks merged → {output_csv}")

        # TemporaryDirectory cleans up chunk files automatically.