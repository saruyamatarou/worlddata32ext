import json
import math
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
COUNTRIES_DIR = BASE / "countries"
OUT_DIR = BASE / "review_packages"

WKT_RE = re.compile(r"POLYGON\s*\(\((.+)\)\)", re.IGNORECASE)

SVG_W = 1200
SVG_H = 900
SVG_PAD = 40


def parse_rect_wkt(wkt: str):
    m = WKT_RE.match(wkt.strip())
    if not m:
        raise ValueError(f"Invalid WKT: {wkt}")

    pts = []
    for part in m.group(1).split(","):
        xy = part.strip().split()
        if len(xy) != 2:
            raise ValueError(f"Invalid point in WKT: {part}")
        x, y = float(xy[0]), float(xy[1])
        pts.append((x, y))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    return {
        "minx": minx,
        "maxx": maxx,
        "miny": miny,
        "maxy": maxy,
        "cx": (minx + maxx) / 2,
        "cy": (miny + maxy) / 2,
        "width": maxx - minx,
        "height": maxy - miny,
    }


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def norm(v: float, vmin: float, vmax: float) -> float:
    if math.isclose(vmin, vmax):
        return 0.5
    return clamp01((v - vmin) / (vmax - vmin))


def zone_from_norm(xn: float, yn: float):
    zones = []

    if yn >= 0.67:
        zones.append("north")
    elif yn <= 0.33:
        zones.append("south")

    if xn >= 0.67:
        zones.append("east")
    elif xn <= 0.33:
        zones.append("west")

    if 0.4 <= xn <= 0.6 and 0.4 <= yn <= 0.6:
        zones.append("center")

    return zones


def bbox_overlap(a, b):
    x_overlap = max(0.0, min(a["maxx"], b["maxx"]) - max(a["minx"], b["minx"]))
    y_overlap = max(0.0, min(a["maxy"], b["maxy"]) - max(a["miny"], b["miny"]))
    return x_overlap, y_overlap


def build_relations(rows):
    for row in rows:
        row["relations"] = {
            "west_of": [],
            "east_of": [],
            "north_of": [],
            "south_of": [],
            "overlaps": []
        }

    for i, a in enumerate(rows):
        ga = a["_geom"]
        for j, b in enumerate(rows):
            if i == j:
                continue
            gb = b["_geom"]

            if ga["cx"] < gb["cx"]:
                a["relations"]["west_of"].append(b["id"])
            elif ga["cx"] > gb["cx"]:
                a["relations"]["east_of"].append(b["id"])

            if ga["cy"] > gb["cy"]:
                a["relations"]["north_of"].append(b["id"])
            elif ga["cy"] < gb["cy"]:
                a["relations"]["south_of"].append(b["id"])

            ox, oy = bbox_overlap(ga, gb)
            if ox > 0 and oy > 0:
                a["relations"]["overlaps"].append({
                    "id": b["id"],
                    "x_overlap": round(ox, 6),
                    "y_overlap": round(oy, 6)
                })


def shorten_name(name: str, max_len: int = 18) -> str:
    if len(name) <= max_len:
        return name
    return name[:max_len - 1] + "…"


def make_svg(iso: str, rows, country_bbox):
    minx = country_bbox["minx"]
    maxx = country_bbox["maxx"]
    miny = country_bbox["miny"]
    maxy = country_bbox["maxy"]

    span_x = max(maxx - minx, 1e-9)
    span_y = max(maxy - miny, 1e-9)

    def sx(x):
        return SVG_PAD + (x - minx) / span_x * (SVG_W - SVG_PAD * 2)

    def sy(y):
        # SVGは上が小さい値なので反転
        return SVG_PAD + (maxy - y) / span_y * (SVG_H - SVG_PAD * 2)

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_W}" height="{SVG_H}" viewBox="0 0 {SVG_W} {SVG_H}">')
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{SVG_PAD}" y="26" font-size="20" font-family="Arial, sans-serif">{iso} review map</text>')
    parts.append(f'<text x="{SVG_PAD}" y="48" font-size="12" font-family="Arial, sans-serif">bbox=({minx:.6f}, {miny:.6f}) - ({maxx:.6f}, {maxy:.6f})</text>')

    # 枠
    parts.append(
        f'<rect x="{SVG_PAD}" y="{SVG_PAD}" width="{SVG_W - SVG_PAD * 2}" height="{SVG_H - SVG_PAD * 2}" '
        f'fill="none" stroke="#999" stroke-width="1"/>'
    )

    for row in rows:
        g = row["_geom"]
        x = sx(g["minx"])
        y = sy(g["maxy"])
        w = max(1, sx(g["maxx"]) - sx(g["minx"]))
        h = max(1, sy(g["miny"]) - sy(g["maxy"]))

        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="none" stroke="black" stroke-width="2"/>'
        )

        label_x = x + 4
        label_y = y + 16
        label = f'{row["id"]} | {shorten_name(row.get("name", ""))}'
        parts.append(
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" font-size="12" font-family="Arial, sans-serif">{escape_xml(label)}</text>'
        )

        cx = sx(g["cx"])
        cy = sy(g["cy"])
        parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="2.5" fill="black"/>')

    parts.append("</svg>")
    return "\n".join(parts)


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
    )


def build_package(rows):
    for row in rows:
        row["_geom"] = parse_rect_wkt(row["geometry_wkt"])

    minx = min(r["_geom"]["minx"] for r in rows)
    maxx = max(r["_geom"]["maxx"] for r in rows)
    miny = min(r["_geom"]["miny"] for r in rows)
    maxy = max(r["_geom"]["maxy"] for r in rows)

    country_bbox = {
        "minx": minx,
        "maxx": maxx,
        "miny": miny,
        "maxy": maxy
    }

    for row in rows:
        g = row["_geom"]
        xn = norm(g["cx"], minx, maxx)
        yn = norm(g["cy"], miny, maxy)

        row["_relative_position"] = {
            "x_norm": round(xn, 4),
            "y_norm": round(yn, 4),
            "zone": zone_from_norm(xn, yn)
        }

    build_relations(rows)

    areas = []
    for row in rows:
        g = row["_geom"]
        areas.append({
            "id": row.get("id"),
            "area_id": row.get("area_id"),
            "area_seq_in_source": row.get("area_seq_in_source"),
            "name": row.get("name"),
            "name_en": row.get("name_en"),
            "reason": row.get("reason"),
            "geometry_wkt": row.get("geometry_wkt"),
            "bbox": {
                "minx": round(g["minx"], 6),
                "maxx": round(g["maxx"], 6),
                "miny": round(g["miny"], 6),
                "maxy": round(g["maxy"], 6)
            },
            "centroid": {
                "x": round(g["cx"], 6),
                "y": round(g["cy"], 6)
            },
            "size": {
                "width": round(g["width"], 6),
                "height": round(g["height"], 6)
            },
            "relative_position": row["_relative_position"],
            "relations": row["relations"]
        })

    iso = rows[0].get("iso_a3", "UNKNOWN") if rows else "UNKNOWN"

    return {
        "iso_a3": iso,
        "area_count": len(rows),
        "country_bbox": {k: round(v, 6) for k, v in country_bbox.items()},
        "areas": areas
    }, country_bbox


def process_country(path: Path):
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path.name}: expected non-empty list")

    package, country_bbox = build_package(rows)
    iso = package["iso_a3"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUT_DIR / f"{iso}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(package, f, ensure_ascii=False, indent=2)

    svg = make_svg(iso, rows, country_bbox)
    svg_path = OUT_DIR / f"{iso}.svg"
    with svg_path.open("w", encoding="utf-8") as f:
        f.write(svg)

    return iso, json_path, svg_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for path in sorted(COUNTRIES_DIR.glob("*.json")):
        iso, _, _ = process_country(path)
        count += 1
        print(f"built: {iso}")

    print(f"done: {count} countries")


if __name__ == "__main__":
    main()