"""Build the math prompt pool (a CSV with a `question` column) from OpenMathInstruct-1.

    python scripts/prepare_math_data.py --out data/openmath_100k.csv --n 100000
"""
import argparse
import os

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/openmath_100k.csv")
    ap.add_argument("--n", type=int, default=100000, help="number of rows to keep")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset("nvidia/OpenMathInstruct-1", data_files="correct_solutions/train.jsonl", split="train")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ds.to_csv(args.out)
    print(f"wrote {len(ds)} rows to {args.out}")


if __name__ == "__main__":
    main()
