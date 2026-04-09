#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Merge per-metric tractmeasures CSV files.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input per-metric tractmeasures CSV files")
    ap.add_argument("--output", required=True, help="Output merged tractmeasures CSV")
    ap.add_argument("--delete-inputs", action="store_true", help="Delete input CSVs after successful merge")
    args = ap.parse_args()

    base_cols = ["subjectID", "structureID", "nodeID"]
    coord_cols = ["x_coords", "y_coords", "z_coords"]
    protected = set(base_cols + coord_cols)

    merged = None

    for csv_path in args.inputs:
        if not os.path.exists(csv_path):
            print(f"[WARN] Missing metric CSV: {csv_path}", file=sys.stderr)
            continue

        df = pd.read_csv(csv_path)

        missing = [c for c in base_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        if merged is None:
            merged = df.copy()
        else:
            metric_cols = [c for c in df.columns if c not in protected]
            df_small = df[base_cols + metric_cols]
            merged = merged.merge(df_small, on=base_cols, how="outer")

    if merged is None:
        raise SystemExit("No valid per-metric tractmeasures CSVs found to merge.")

    metric_cols_all = [c for c in merged.columns if c not in protected]
    ordered_cols = base_cols + metric_cols_all

    for c in coord_cols:
        if c in merged.columns:
            ordered_cols.append(c)

    merged = merged[ordered_cols]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"[INFO] Saved merged tract measures -> {args.output}")

    if args.delete_inputs:
        for csv_path in args.inputs:
            try:
                os.remove(csv_path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
