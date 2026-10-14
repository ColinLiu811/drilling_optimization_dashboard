"""Download the Utah FORGE 16A(78)-32 drilling data and build the binned dataset.

Usage:
    python scripts/prepare_forge.py                   # download + process
    python scripts/prepare_forge.py --raw path.csv    # use an existing download
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drillopt.data import DATA_DIR, FORGE_URL, PROCESSED_PATH, prepare  # noqa: E402


def download(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {FORGE_URL}\n  -> {dest} (~120 MB)")
    urllib.request.urlretrieve(FORGE_URL, dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, help="existing raw 10 s CSV")
    ap.add_argument("--interval", type=float, default=1.0, help="depth bin size, ft")
    ap.add_argument("--out", type=Path, default=PROCESSED_PATH)
    args = ap.parse_args()

    raw = args.raw or DATA_DIR / "raw" / "16A78-32_time_data_10s_intervals.csv"
    if not raw.exists():
        download(raw)

    df = prepare(raw, args.interval)
    df = df.round({c: 3 for c in df.select_dtypes("number").columns})
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} depth bins ({df.depth.min():.0f}-{df.depth.max():.0f} ft) to {args.out}")


if __name__ == "__main__":
    main()
