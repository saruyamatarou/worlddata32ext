import json
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE / "input" / "country_area_with_area_id.json"
OUTPUT_DIR = BASE / "countries"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list")

    grouped = defaultdict(list)

    for row in data:
        iso = row.get("iso_a3")
        if not iso:
            iso = "UNKNOWN"
        grouped[iso].append(row)

    for iso, rows in grouped.items():
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                str(r.get("iso_a3", "")),
                int(r.get("area_seq_in_source", 0)),
                int(r.get("area_id", 0))
            )
        )
        out_path = OUTPUT_DIR / f"{iso}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

    print(f"countries written: {len(grouped)}")
    total = sum(len(v) for v in grouped.values())
    print(f"rows written: {total}")

if __name__ == "__main__":
    main()