"""
show_predictions.py — Human-readable species predictions from submission.csv

Usage:
    python scripts/show_predictions.py                          # top detections summary
    python scripts/show_predictions.py --file H02_20230420_074000  # single recording
    python scripts/show_predictions.py --threshold 0.3         # custom threshold
"""

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission",  default="submission.csv")
    parser.add_argument("--logdir",      default="logdir")
    parser.add_argument("--taxonomy",    default="D:/birdclef_data/birdclef-2025/taxonomy.csv")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Probability threshold to call a species 'present'")
    parser.add_argument("--file",        default=None,
                        help="Show predictions for one recording (e.g. H02_20230420_074000)")
    parser.add_argument("--top",         type=int, default=20,
                        help="Show top-N most detected species overall")
    args = parser.parse_args()

    # ── Load submission ───────────────────────────────────────────────────
    df = pd.read_csv(args.submission, index_col=0)
    print(f"Loaded submission: {len(df):,} windows × {len(df.columns)} species\n")

    # ── Load label mapping ────────────────────────────────────────────────
    # Find label_to_idx.json in logdir
    manifest_paths = list(Path(args.logdir).glob("*/label_to_idx.json"))
    if not manifest_paths:
        print("ERROR: label_to_idx.json not found in logdir/")
        return
    with open(manifest_paths[0]) as f:
        label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # ── Load taxonomy (species codes → common names) ──────────────────────
    tax = pd.read_csv(args.taxonomy)
    tax["primary_label"] = tax["primary_label"].astype(str)
    label_to_name = dict(zip(tax["primary_label"], tax["common_name"]))
    label_to_sci  = dict(zip(tax["primary_label"], tax["scientific_name"]))

    # ── Rename columns: species_0 → actual species code ──────────────────
    col_map = {f"species_{i}": idx_to_label[i] for i in range(len(df.columns))}
    df.rename(columns=col_map, inplace=True)

    top_k = args.top if args.top else 3

    # ── Mode 1: Single recording ──────────────────────────────────────────
    if args.file:
        rows = df[df.index.str.startswith(args.file)]
        if rows.empty:
            print(f"No rows found for recording: {args.file}")
            return

        print(f"{'='*65}")
        print(f"  Recording: {args.file}")
        print(f"  Windows: {len(rows)}  |  Showing top {top_k} species per window")
        print(f"{'='*65}")

        for row_id, probs in rows.iterrows():
            t       = float(row_id.split("_")[-1])
            top     = probs.sort_values(ascending=False).head(top_k)
            print(f"\n  [{t:.0f}s – {t+5:.0f}s]")
            for code, prob in top.items():
                name = label_to_name.get(code, code)
                sci  = label_to_sci.get(code, "")
                bar  = "█" * int(prob * 500) or "░"
                print(f"    {prob:.4f}  {bar:<8}  {name:<35} ({sci})")
        return

    # ── Mode 2: Overall summary — most commonly top-ranked species ────────
    print(f"Top {top_k} species per window — most frequently predicted across all soundscapes\n")

    # For each window, find the top-k species; count how often each appears
    from collections import Counter
    counter = Counter()
    for _, probs in df.iterrows():
        top = probs.sort_values(ascending=False).head(top_k).index.tolist()
        counter.update(top)

    total_windows = len(df)
    print(f"{'Rank':<5} {'Species':<35} {'Scientific':<30} {'Windows':>8} {'% of windows':>13}")
    print("-" * 95)
    for rank, (code, count) in enumerate(counter.most_common(20), 1):
        name = label_to_name.get(code, code)
        sci  = label_to_sci.get(code, "")
        pct  = count / total_windows * 100
        print(f"{rank:<5} {name:<35} {sci:<30} {count:>8,} {pct:>12.1f}%")

    print(f"\nTotal 5-second windows: {total_windows:,}")


if __name__ == "__main__":
    main()
