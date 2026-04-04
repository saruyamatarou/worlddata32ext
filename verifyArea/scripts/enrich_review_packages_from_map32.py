import json
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path(__file__).resolve().parent.parent

MAP32_PATH = BASE / "input" / "map32.bin"
REVIEW_DIR = BASE / "review_packages"
OUT_INDEX = REVIEW_DIR / "_area_map32_summary.json"

WIDTH = 8640
HEIGHT = 4320
CELL_BYTES = 4
DLON = 360.0 / WIDTH
DLAT = 180.0 / HEIGHT

LANDCOVER = {
    0: "UNKNOWN",
    1: "TREE",
    2: "SHRUB",
    3: "GRASS",
    4: "CROPLAND",
    5: "BUILT",
    6: "BARE",
    7: "SNOW_ICE",
    8: "WATER",
    9: "WETLAND",
    10: "MANGROVE",
    11: "MOSS_LICHEN",
    12: "OCEAN",
}

T2 = {
    0: "POLAR",
    1: "COLD",
    2: "TEMPERATE",
    3: "TROPICAL",
}

E2 = {
    0: "LOW",
    1: "MID",
    2: "HIGH",
    3: "ULTRA",
}

P2 = {
    0: "ARID",
    1: "LOW",
    2: "MID",
    3: "WET",
}

FLAG_BITS = [
    "water",
    "built",
    "cropland",
    "reef",
    "forest",
    "ice",
]


def new_area_stat():
    return {
        "cell_count": 0,
        "landcover_counts": Counter(),
        "t2_counts": Counter(),
        "e2_counts": Counter(),
        "p2_counts": Counter(),
        "flag_counts": Counter(),
        "x_sum": 0.0,
        "y_sum": 0.0,
        "min_x": None,
        "max_x": None,
        "min_y": None,
        "max_y": None,
    }


def update_bbox(stat, x, y):
    if stat["min_x"] is None or x < stat["min_x"]:
        stat["min_x"] = x
    if stat["max_x"] is None or x > stat["max_x"]:
        stat["max_x"] = x
    if stat["min_y"] is None or y < stat["min_y"]:
        stat["min_y"] = y
    if stat["max_y"] is None or y > stat["max_y"]:
        stat["max_y"] = y


def ratio_dict(counter, denom):
    if denom <= 0:
        return {}
    out = {}
    for key, count in counter.most_common():
        out[key] = round(count / denom, 4)
    return out


def dominant_name(counter):
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def xy_to_lonlat_center(x, y):
    lon = -180.0 + (x + 0.5) * DLON
    lat = 90.0 - (y + 0.5) * DLAT
    return lon, lat


def decode_cell(b0, b1, b2, b3):
    lo16 = b0 | (b1 << 8)
    area_u16 = b2 | (b3 << 8)
    area_id = area_u16 & 0x03FF

    wc_dom = lo16 & 0x0F
    t2 = (lo16 >> 4) & 0x03
    e2 = (lo16 >> 6) & 0x03
    p2 = (lo16 >> 8) & 0x03
    flags6 = (lo16 >> 10) & 0x3F

    return area_id, wc_dom, t2, e2, p2, flags6


def build_area_summary_index():
    stats = defaultdict(new_area_stat)

    total_cells = WIDTH * HEIGHT
    expected_size = total_cells * CELL_BYTES
    actual_size = MAP32_PATH.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"map32.bin size mismatch: expected {expected_size}, got {actual_size}"
        )

    with MAP32_PATH.open("rb") as f:
        for y in range(HEIGHT):
            row = f.read(WIDTH * CELL_BYTES)
            if len(row) != WIDTH * CELL_BYTES:
                raise ValueError(f"Short read at row {y}")

            for x in range(WIDTH):
                i = x * 4
                b0 = row[i]
                b1 = row[i + 1]
                b2 = row[i + 2]
                b3 = row[i + 3]

                area_id, wc_dom, t2, e2, p2, flags6 = decode_cell(b0, b1, b2, b3)

                if area_id == 0:
                    continue

                stat = stats[area_id]
                stat["cell_count"] += 1
                stat["landcover_counts"][LANDCOVER.get(wc_dom, f"LC_{wc_dom}")] += 1
                stat["t2_counts"][T2.get(t2, f"T2_{t2}")] += 1
                stat["e2_counts"][E2.get(e2, f"E2_{e2}")] += 1
                stat["p2_counts"][P2.get(p2, f"P2_{p2}")] += 1

                for bit_index, flag_name in enumerate(FLAG_BITS):
                    if flags6 & (1 << bit_index):
                        stat["flag_counts"][flag_name] += 1

                stat["x_sum"] += x
                stat["y_sum"] += y
                update_bbox(stat, x, y)

    summary = {}

    for area_id, stat in stats.items():
        cell_count = stat["cell_count"]
        cx = stat["x_sum"] / cell_count
        cy = stat["y_sum"] / cell_count
        lon, lat = xy_to_lonlat_center(cx, cy)

        summary[str(area_id)] = {
            "cell_count": cell_count,
            "dominant_landcover": dominant_name(stat["landcover_counts"]),
            "dominant_t2": dominant_name(stat["t2_counts"]),
            "dominant_e2": dominant_name(stat["e2_counts"]),
            "dominant_p2": dominant_name(stat["p2_counts"]),
            "landcover_ratio": ratio_dict(stat["landcover_counts"], cell_count),
            "temperature_ratio": ratio_dict(stat["t2_counts"], cell_count),
            "elevation_ratio": ratio_dict(stat["e2_counts"], cell_count),
            "precip_ratio": ratio_dict(stat["p2_counts"], cell_count),
            "flags_ratio": ratio_dict(stat["flag_counts"], cell_count),
            "cell_bbox_xy": {
                "min_x": stat["min_x"],
                "max_x": stat["max_x"],
                "min_y": stat["min_y"],
                "max_y": stat["max_y"],
            },
            "cell_centroid": {
                "x": round(cx, 3),
                "y": round(cy, 3),
                "lon": round(lon, 6),
                "lat": round(lat, 6),
            },
        }

    return summary


def enrich_review_packages(area_summary):
    for path in sorted(REVIEW_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue

        with path.open("r", encoding="utf-8") as f:
            pkg = json.load(f)

        if not isinstance(pkg, dict) or "areas" not in pkg:
            continue

        changed = False

        for area in pkg["areas"]:
            area_id = area.get("area_id")
            if area_id is None:
                continue

            summary = area_summary.get(str(area_id))
            if summary is None:
                area["map32_summary"] = {
                    "cell_count": 0,
                    "status": "no_cells_found_in_map32"
                }
            else:
                area["map32_summary"] = summary

            changed = True

        if changed:
            with path.open("w", encoding="utf-8") as f:
                json.dump(pkg, f, ensure_ascii=False, indent=2)


def main():
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    area_summary = build_area_summary_index()

    with OUT_INDEX.open("w", encoding="utf-8") as f:
        json.dump(area_summary, f, ensure_ascii=False, indent=2)

    enrich_review_packages(area_summary)

    print(f"area summaries: {len(area_summary)}")
    print("review packages enriched")


if __name__ == "__main__":
    main()