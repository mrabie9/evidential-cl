import csv
import sys
from pathlib import Path

FILES = [Path(p) for p in sys.argv[1:]]
if not FILES:
    print("Usage: print_bwt_fwt.py <file1.csv> [file2.csv ...]", file=sys.stderr)
    sys.exit(1)

for path in FILES:
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        continue
    print(f"\n=== {path} ===")
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print("(empty)")
        continue
    cols = list(rows[0].keys())
    col_widths = {c: max(len(c), max(len(r[c]) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(col_widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(row[c].ljust(col_widths[c]) for c in cols))
