"""Download a stratified subset of Fakeddit images from image URLs in the TSV files.

Based on the official Fakeddit image_downloader.py, but instead of downloading
all ~100GB of images it samples N rows (balanced across labels) and downloads
only those, in parallel, skipping dead links.

Usage:
    python src/download_subset.py multimodal_train.tsv --n 10000 --out data/images

The sampled rows are also saved as <input>_subset.tsv so the dataset code
only ever sees rows whose images actually downloaded successfully.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (fakeddit-subset-downloader)"}
TIMEOUT = 10


def download_one(row, out_dir):
    url = row["image_url"]
    ext = os.path.splitext(str(url).split("?")[0])[1] or ".jpg"
    path = os.path.join(out_dir, f"{row['id']}{ext}")
    if os.path.exists(path):
        return row["id"], path
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        # Reddit returns an HTML error page (not an image) for dead links
        if "image" not in r.headers.get("Content-Type", ""):
            return row["id"], None
        with open(path, "wb") as f:
            f.write(r.content)
        return row["id"], path
    except Exception:
        return row["id"], None


def main():
    ap = argparse.ArgumentParser(description="Fakeddit subset image downloader")
    ap.add_argument("tsv", help="Fakeddit tsv file (e.g. multimodal_train.tsv)")
    ap.add_argument("--n", type=int, default=10000, help="total samples to keep")
    ap.add_argument("--label-col", default="2_way_label",
                    help="column to stratify by (2_way_label, 3_way_label, 6_way_label)")
    ap.add_argument("--out", default="data/images", help="output image directory")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.tsv, sep="\t")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df[df["hasImage"] == True]
    df = df[df["image_url"].notna() & (df["image_url"] != "") & (df["image_url"] != "nan")]

    # stratified sample: equal count per class
    per_class = args.n // df[args.label_col].nunique()
    parts = [g.sample(min(len(g), per_class), random_state=args.seed)
             for _, g in df.groupby(args.label_col)]
    sampled = pd.concat(parts).reset_index(drop=True)
    print(f"Sampled {len(sampled)} rows "
          f"({dict(sampled[args.label_col].value_counts())}) — downloading...")

    os.makedirs(args.out, exist_ok=True)
    ok_ids = set()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(download_one, row, args.out) for _, row in sampled.iterrows()]
        for i, fut in enumerate(as_completed(futures), 1):
            sid, path = fut.result()
            if path:
                ok_ids.add(sid)
            if i % 500 == 0:
                print(f"{i}/{len(futures)} attempted, {len(ok_ids)} succeeded")

    kept = sampled[sampled["id"].isin(ok_ids)]
    out_tsv = os.path.splitext(args.tsv)[0] + "_subset.tsv"
    kept.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nDone: {len(kept)}/{len(sampled)} images downloaded to {args.out}")
    print(f"Matching metadata saved to {out_tsv}")


if __name__ == "__main__":
    main()
