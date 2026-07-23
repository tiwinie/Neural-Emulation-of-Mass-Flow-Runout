#!/usr/bin/env python3
import os
import json
import shutil
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="data_split")
    ap.add_argument("--move", action="store_true")
    args = ap.parse_args()

    with open(args.json) as f:
        splits = json.load(f)

    print(f"Found splits: {list(splits.keys())}")
    op = shutil.move if args.move else shutil.copy2
    verb = "Moved" if args.move else "Copied"
    total_ok, total_missing = 0, 0

    for split_name, filenames in splits.items():
        split_dir = os.path.join(args.out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        ok, missing = 0, []
        for fname in filenames:
            fname = os.path.basename(fname)
            src = os.path.join(args.data_dir, fname)
            dst = os.path.join(split_dir, fname)
            if not os.path.exists(src):
                missing.append(fname)
                continue
            op(src, dst)
            ok += 1
        print(f"\n[{split_name}] {verb.lower()} {ok}/{len(filenames)} files -> {split_dir}")
        if missing:
            print(f"  WARNING: {len(missing)} file(s) listed in JSON but not found in {args.data_dir}")
            for m in missing[:10]:
                print(f"      - {m}")
        total_ok += ok
        total_missing += len(missing)

    print(f"\nDone. {verb} {total_ok} files total. {total_missing} missing overall.")


if __name__ == "__main__":
    main()
