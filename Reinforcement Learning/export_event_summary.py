"""
export_event_summary.py
Tallies labeled game events from rugby_detection.csv and writes event_summary.csv.

Run from the Reinforcement Learning/ directory:
    python export_event_summary.py

Outputs: event_summary.csv  (columns: Event, Count)
"""

import pandas as pd
import os

IN_PATH  = os.path.join("data", "rugby_detection.csv")
OUT_PATH = "event_summary.csv"

EVENT_LABELS = {
    "Pass":      "Pass",
    "Carry":     "Carry",
    "Kick":      "Kick",
    "Try":       "Try",
    "Turnover":  "Turnover",
}


def main():
    df = pd.read_csv(IN_PATH, low_memory=False)

    if "ball_action" not in df.columns:
        print(f"Error: 'ball_action' column not found in {IN_PATH}")
        return

    counts = (
        df["ball_action"]
        .dropna()
        .astype(str)
        .str.strip()
        .value_counts()
    )

    rows = []
    for raw, label in EVENT_LABELS.items():
        count = int(counts.get(raw, 0))
        if count > 0:
            rows.append({"Event": label, "Count": count})

    # Include any unlisted event types found in the data
    for event, count in counts.items():
        if event and event not in EVENT_LABELS and event.lower() not in ("nan", "none", ""):
            rows.append({"Event": event.title(), "Count": int(count)})

    if not rows:
        print("No labeled events found. Check ball_action column values.")
        return

    out = pd.DataFrame(rows).sort_values("Count", ascending=False)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH}:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
