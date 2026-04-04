import json
import math
import re
from collections import defaultdict
from pathlib import Path

import fiona
import numpy as np
import rasterio
from rasterio import features

try:
    from scipy.ndimage import distance_transform_edt
except Exception as e:
    raise SystemExit("scipy が必要です: pip install scipy") from e


ALIAS_TO_ISO = {
    "AX": "ALA",
    "BF": "BFA",
    "BZ": "BLZ",
    "CR": "CRI",
    "DGA": "IOT",
    "DM": "DMA",
    "EH": "ESH",
    "FK": "FLK",
    "GL": "GRL",
    "GY": "GUY",
    "HT": "HTI",
    "JE": "JEY",
    "KG": "KGZ",
    "LT": "LTU",
    "LV": "LVA",
    "NR": "NRU",
    "SB": "SLB",
    "SH": "SHN",
    "SM": "SMR",
    "SOK": "ATA",   # South Orkney Islands -> Antarctica
    "SRP": "BIH",   # Republika Srpska -> Bosnia and Herzegovina
    "TT": "TTO",
    "TW": "TWN",
    "VC": "VCT",
}

INVALID_CODE_VALUES = {"", "NULL", "-99"}


# country_area_with_area_id.json のうち、現状 area_legend.json に明示的に持ち込んでいない項目。
# これらは従来どおり area_legend へはコピーしない。
AREA_LEGEND_EXCLUDED_SOURCE_FIELDS = {
    "id",            # source_id に正規化して保持
    "geometry_wkt",  # 既存仕様では area_legend へは入れない
    "area_id",       # id_to_meta のキーとして保持
}

# load_country_areas 内部で付与する管理用フィールド
AREA_INTERNAL_FIELDS = {
    "bounds",
    "geoms",
    "_passthrough",
}


def load_grid_spec(path: str):
    spec = json.load(open(path, "r", encoding="utf-8"))
    H = int(spec["height"])
    W = int(spec["width"])
    a, b, c, d, e, f = spec["transform"]
    transform = rasterio.Affine(a, b, c, d, e, f)
    crs = spec.get("crs", "EPSG:4326")
    return spec, H, W, transform, crs


def read_mask_u8_tif(path: str) -> np.ndarray:
    with rasterio.open(path) as ds:
        a = ds.read(1)
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    return a


def _get_prop_ci(props: dict, key: str):
    if props is None:
        return None
    lk = key.lower()
    for k, v in props.items():
        if str(k).lower() == lk:
            return v
    return None


def _norm_code(v):
    if v is None:
        return None
    s = str(v).strip()
    if s in INVALID_CODE_VALUES:
        return None
    return s


def _pick_first_valid_prop(props: dict, keys: list[str]):
    for key in keys:
        v = _norm_code(_get_prop_ci(props, key))
        if v is not None:
            return v
    return None


def load_name_ja_map(ne_names_shp: str):
    if not ne_names_shp or not Path(ne_names_shp).exists():
        return {}

    name_ja_map = {}
    ja_keys_candidates = [
        "NAME_JA", "NAME_JA1", "NAME_JA2", "NAME_JA3", "NAME_JA4",
        "NAME_JA5", "NAME_JA6", "NAME_JA7", "NAME_JA8", "NAME_JA9", "NAME_JA10",
        "NAME_JAP", "NAME_JAPAN", "NAME_JP", "NAME_JPN",
        "name_ja", "name_jp",
    ]

    with fiona.open(ne_names_shp, "r") as src:
        for feat in src:
            prop = feat["properties"] or {}
            key = _pick_first_valid_prop(prop, ["ADM0_A3", "ISO_A3", "SOV_A3", "GU_A3", "SU_A3", "BRK_A3"])
            if key is None:
                continue

            ja = None
            for jk in ja_keys_candidates:
                v = _norm_code(_get_prop_ci(prop, jk))
                if v is not None:
                    ja = v
                    break

            if ja:
                name_ja_map[key] = ja

    return name_ja_map


def load_country_legend_name_overrides(path: str | None):
    """
    country_legend.json の id_to_meta から iso_a3 -> {name_ja, name_en} を作る。
    数値IDではなく iso_a3 で引くので、生成し直して country_id が変わっても使える。
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"country legend source not found: {p}")

    try:
        obj = json.load(open(p, "r", encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"country legend source の JSON 読込に失敗しました: {p}\n{e}") from e

    id_to_meta = obj.get("id_to_meta")
    if not isinstance(id_to_meta, dict):
        raise SystemExit(f"country legend source に id_to_meta がありません: {p}")

    out = {}
    for _id, meta in id_to_meta.items():
        if not isinstance(meta, dict):
            continue
        iso = _norm_code(meta.get("iso_a3"))
        if iso is None or iso == "NONE":
            continue
        row = {}
        name_ja = _norm_code(meta.get("name_ja"))
        name_en = _norm_code(meta.get("name_en"))
        if name_ja is not None:
            row["name_ja"] = name_ja
        if name_en is not None:
            row["name_en"] = name_en
        if row:
            out[iso] = row
    return out


def build_country_ids(shp_path: str, ne_names_shp: str | None = None, country_legend_source: str | None = None):
    name_ja_map = load_name_ja_map(ne_names_shp)
    legend_name_overrides = load_country_legend_name_overrides(country_legend_source)

    records = []
    iso_set = set()
    first_props_by_iso = {}

    with fiona.open(shp_path, "r") as src:
        for feat in src:
            prop = feat["properties"] or {}

            iso_a3 = _pick_first_valid_prop(prop, ["ISO_A3", "ADM0_A3", "SOV_A3", "GU_A3", "SU_A3", "BRK_A3"])
            if iso_a3 is None:
                continue

            iso_set.add(iso_a3)
            records.append((iso_a3, prop, feat["geometry"]))

            if iso_a3 not in first_props_by_iso:
                first_props_by_iso[iso_a3] = prop

    iso_list = sorted(list(iso_set))
    iso_to_id = {iso: i + 1 for i, iso in enumerate(iso_list)}  # 1..N (内部用)

    id_to_meta = {"0": {"iso_a3": "NONE", "name_en": "No country", "name_ja": "無国籍"}}
    for iso_a3, cid in iso_to_id.items():
        p = first_props_by_iso.get(iso_a3, {}) or {}

        iso_n3 = _pick_first_valid_prop(p, ["ISO_N3", "ISO3_N", "ADM0_ISO_N"])
        ne_id = _pick_first_valid_prop(p, ["NE_ID", "ne_id", "FEATUREID"])

        name_en = (
            _norm_code(_get_prop_ci(p, "NAME"))
            or _norm_code(_get_prop_ci(p, "ADMIN"))
            or _norm_code(_get_prop_ci(p, "SOVEREIGNT"))
            or _norm_code(_get_prop_ci(p, "NAME_EN"))
        )

        name_ja = name_ja_map.get(iso_a3)
        if not name_ja:
            name_ja = _norm_code(_get_prop_ci(p, "NAME_JA")) or _norm_code(_get_prop_ci(p, "name_ja"))

        ov = legend_name_overrides.get(iso_a3, {})
        if "name_en" in ov:
            name_en = ov["name_en"]
        if "name_ja" in ov:
            name_ja = ov["name_ja"]

        id_to_meta[str(cid)] = {
            "iso_a3": iso_a3,
            "iso_n3": iso_n3,
            "ne_id": int(ne_id) if isinstance(ne_id, (int, np.integer)) or (isinstance(ne_id, str) and ne_id.isdigit()) else ne_id,
            "name_en": name_en,
            "name_ja": name_ja,
        }

    return records, iso_to_id, id_to_meta


def rasterize_countries(records, iso_to_id, out_shape, transform) -> np.ndarray:
    shapes = []
    for iso, _props, geom in records:
        if geom is None:
            continue
        cid = iso_to_id.get(iso)
        if cid is None:
            continue
        shapes.append((geom, int(cid)))

    arr = features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint16,
        all_touched=False,
    )
    return arr


def fill_nearshore(id_u16: np.ndarray, ocean_u8: np.ndarray, buffer_cells: int) -> np.ndarray:
    land = id_u16 > 0
    dist, (iy, ix) = distance_transform_edt(~land, return_indices=True)
    out = id_u16.copy()

    ocean = ocean_u8 > 0
    target = ocean & (out == 0) & (dist <= float(buffer_cells))
    out[target] = out[iy[target], ix[target]]
    return out


def fill_inland_holes(id_u16: np.ndarray, ocean_u8: np.ndarray, max_cells: int = 3) -> np.ndarray:
    land_or_inland = ocean_u8 == 0
    hole = land_or_inland & (id_u16 == 0)
    if not np.any(hole):
        return id_u16

    seed = id_u16 > 0
    dist, (iy, ix) = distance_transform_edt(~seed, return_indices=True)
    out = id_u16.copy()
    target = hole & (dist <= float(max_cells))
    out[target] = out[iy[target], ix[target]]
    return out


def parse_rect_bounds_from_wkt(wkt: str):
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", wkt)]
    if len(nums) < 8 or len(nums) % 2 != 0:
        raise ValueError(f"invalid rectangle WKT: {wkt[:120]}")
    pts = list(zip(nums[0::2], nums[1::2]))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    ymin = max(-90.0, ymin)
    ymax = min(90.0, ymax)
    return xmin, ymin, xmax, ymax


def _box_geom(xmin: float, ymin: float, xmax: float, ymax: float):
    return {
        "type": "Polygon",
        "coordinates": [[
            [xmin, ymax],
            [xmax, ymax],
            [xmax, ymin],
            [xmin, ymin],
            [xmin, ymax],
        ]],
    }


def rect_bounds_to_geoms(xmin: float, ymin: float, xmax: float, ymax: float):
    """
    [-180, 180] を跨ぐ長方形を必要に応じて2つに分割する。
    country_area.json は矩形前提。
    """
    if xmax <= 180.0 and xmin >= -180.0:
        return [_box_geom(xmin, ymin, xmax, ymax)]

    if xmax > 180.0 and xmin >= -180.0:
        east2 = xmax - 360.0
        return [
            _box_geom(xmin, ymin, 180.0, ymax),
            _box_geom(-180.0, ymin, east2, ymax),
        ]

    if xmin < -180.0 and xmax <= 180.0:
        west2 = xmin + 360.0
        return [
            _box_geom(-180.0, ymin, xmax, ymax),
            _box_geom(west2, ymin, 180.0, ymax),
        ]

    xmin = max(-180.0, xmin)
    xmax = min(180.0, xmax)
    return [_box_geom(xmin, ymin, xmax, ymax)]


def load_country_areas(path: str):
    rows = json.load(open(path, "r", encoding="utf-8"))
    out_rows = []
    seq_by_prefix = defaultdict(int)

    for source_row_index, rec in enumerate(rows, start=1):
        source_id = str(rec["id"]).strip()
        src_prefix = source_id.split("-")[0]
        iso_a3 = ALIAS_TO_ISO.get(src_prefix, src_prefix)

        seq_by_prefix[src_prefix] += 1
        area_id = len(out_rows) + 1

        xmin, ymin, xmax, ymax = parse_rect_bounds_from_wkt(rec["geometry_wkt"])
        geoms = rect_bounds_to_geoms(xmin, ymin, xmax, ymax)

        passthrough = {}
        for k, v in rec.items():
            if k in AREA_LEGEND_EXCLUDED_SOURCE_FIELDS:
                continue
            if k in {
                "source_row_index",
                "iso_a3_source",
                "iso_a3",
                "area_seq_in_source",
                "name",
                "name_en",
                "reason",
            }:
                # 既存項目は area_meta 側で個別に入れるので重複コピーしない
                continue
            passthrough[k] = v

        out_rows.append({
            "area_id": area_id,
            "source_row_index": source_row_index,
            "source_id": source_id,
            "iso_a3_source": src_prefix,
            "iso_a3": iso_a3,
            "area_seq_in_source": seq_by_prefix[src_prefix],
            "name": rec.get("name"),
            "name_en": rec.get("name_en"),
            "reason": rec.get("reason"),
            "bounds": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            "geoms": geoms,
            "_passthrough": passthrough,
        })

    return out_rows


def rasterize_area_rectangles(area_rows, out_shape, transform):
    shapes = []
    for row in area_rows:
        area_id = int(row["area_id"])
        for geom in row["geoms"]:
            shapes.append((geom, area_id))

    if len(area_rows) > 1023:
        raise SystemExit(f"area id overflow (>1023): count={len(area_rows)}")

    arr = features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint16,
        all_touched=False,
    )
    return arr


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-spec", default="out/grid/grid_spec.json")
    ap.add_argument("--ne-countries", default="data/10m_cultural/ne_10m_admin_0_countries.shp")
    ap.add_argument("--ne-names", default="data/10m_cultural/ne_10m_admin_0_names.shp")
    ap.add_argument("--ocean-mask", default="out/ne/ne_ocean_mask.tif")
    ap.add_argument("--country-area-json", default="country_area_with_area_id.json")
    ap.add_argument("--country-legend-source", default="country_legend.json")

    ap.add_argument("--nearshore-km", type=float, default=50.0)
    ap.add_argument("--cell-km", type=float, default=5.0)


    ap.add_argument("--out-area-u16", default="out/area/area_id.u16")
    ap.add_argument("--out-area-legend", default="out/area/area_legend.json")
    ap.add_argument("--out-country-area-with-id", default="out/area/country_area_with_area_id.json")
    args = ap.parse_args()

    spec, H, W, transform, _crs = load_grid_spec(args.grid_spec)
    ocean = read_mask_u8_tif(args.ocean_mask)
    if ocean.shape != (H, W):
        raise SystemExit(f"ocean mask shape mismatch: {ocean.shape} vs {(H, W)}")

    records, iso_to_country_id, country_id_to_meta = build_country_ids(
        args.ne_countries,
        args.ne_names,
        args.country_legend_source,
    )

    country_base = rasterize_countries(records, iso_to_country_id, (H, W), transform)
    buffer_cells = int(math.ceil(args.nearshore_km / args.cell_km))
    country_map = fill_nearshore(country_base, ocean, buffer_cells=buffer_cells)
    country_map = fill_inland_holes(country_map, ocean, max_cells=3)

    area_rows = load_country_areas(args.country_area_json)

    out_area_rows = []
    for row in area_rows:
        out_row = {
            "area_id": row["area_id"],
            "source_row_index": row["source_row_index"],
            "id": row["source_id"],
            "iso_a3_source": row["iso_a3_source"],
            "iso_a3": row["iso_a3"],
            "area_seq_in_source": row["area_seq_in_source"],
            "name": row["name"],
            "name_en": row["name_en"],
            "reason": row["reason"],
            "geometry_wkt": f'POLYGON (({row["bounds"]["xmin"]} {row["bounds"]["ymax"]}, {row["bounds"]["xmax"]} {row["bounds"]["ymax"]}, {row["bounds"]["xmax"]} {row["bounds"]["ymin"]}, {row["bounds"]["xmin"]} {row["bounds"]["ymin"]}, {row["bounds"]["xmin"]} {row["bounds"]["ymax"]}))',
        }
        out_row.update(row.get("_passthrough", {}))
        out_area_rows.append(out_row)
    Path(args.out_country_area_with_id).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_country_area_with_id, "w", encoding="utf-8") as f:
        json.dump(out_area_rows, f, ensure_ascii=False, indent=2)

    area_raw = rasterize_area_rectangles(area_rows, (H, W), transform)

    area_to_country = np.zeros(len(area_rows) + 1, dtype=np.uint16)
    area_meta = {
        "0": {
            "source_id": "NONE",
            "iso_a3": "NONE",
            "iso_a3_source": "NONE",
            "country_id": 0,
            "area_seq_in_source": 0,
            "name": "無所属",
            "name_en": "No area",
            "reason": None,
            "source_row_index": 0,
        }
    }
    default_area_by_country = {}

    for row in area_rows:
        area_id = int(row["area_id"])
        iso_a3 = row["iso_a3"]
        if iso_a3 not in iso_to_country_id:
            raise SystemExit(
                f"country_area の ISO_A3 を country shp に対応付けできません: "
                f"source_id={row['source_id']} iso_a3={iso_a3}"
            )
        country_id = int(iso_to_country_id[iso_a3])
        area_to_country[area_id] = country_id
        default_area_by_country.setdefault(country_id, area_id)

        country_meta = country_id_to_meta.get(str(country_id), {})
        meta = {
            "source_id": row["source_id"],
            "iso_a3": iso_a3,
            "iso_a3_source": row["iso_a3_source"],
            "country_id": country_id,
            "country_name_ja": country_meta.get("name_ja"),
            "country_name_en": country_meta.get("name_en"),
            "country_iso_n3": country_meta.get("iso_n3"),
            "area_seq_in_source": row["area_seq_in_source"],
            "name": row["name"],
            "name_en": row["name_en"],
            "reason": row["reason"],
            "source_row_index": row["source_row_index"],
        }
        meta.update(row.get("_passthrough", {}))
        area_meta[str(area_id)] = meta

    area_country = area_to_country[area_raw]
    area_land = np.where((area_raw > 0) & (country_base == area_country), area_raw, 0).astype(np.uint16)

    fallback_mask = (country_base > 0) & (area_land == 0)
    if np.any(fallback_mask):
        country_to_default = np.zeros(int(country_base.max()) + 1, dtype=np.uint16)
        for cid, aid in default_area_by_country.items():
            country_to_default[int(cid)] = int(aid)
        area_land[fallback_mask] = country_to_default[country_base[fallback_mask]]

    area_map = fill_nearshore(area_land, ocean, buffer_cells=buffer_cells)
    area_map = fill_inland_holes(area_map, ocean, max_cells=3)

    max_area = int(area_map.max())
    if max_area > 1023:
        raise SystemExit(f"area id overflow (>1023): max={max_area}")

    Path(args.out_area_u16).parent.mkdir(parents=True, exist_ok=True)
    area_map.astype(np.uint16).tofile(args.out_area_u16)

    Path(args.out_area_legend).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_area_legend, "w", encoding="utf-8") as f:
        json.dump(
            {
                "area_bits": 10,
                "area_count": len(area_rows),
                "area_id_max_used": max_area,
                "area_id_capacity": 1023,
                "nearshore_km": args.nearshore_km,
                "cell_km_assumed": args.cell_km,
                "country_alias_to_iso_a3": ALIAS_TO_ISO,
                "country_name_source": args.country_legend_source if args.country_legend_source else "natural_earth",
                "id_to_meta": area_meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[OK] wrote:", args.out_country_area_with_id)
    print("[OK] wrote:", args.out_area_u16, "bytes=", Path(args.out_area_u16).stat().st_size)
    print("[OK] wrote:", args.out_area_legend)


if __name__ == "__main__":
    main()
