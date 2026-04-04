# 08_pack_final_map32.py
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject


def load_grid_spec(path: str):
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    H = int(spec["height"])
    W = int(spec["width"])
    return spec, H, W


def _die(msg: str):
    raise SystemExit(msg)


def require_file(path: str, what: str):
    p = Path(path)
    if not p.exists():
        _die(f"[ERROR] missing {what}: {p}")
    if not p.is_file():
        _die(f"[ERROR] not a file {what}: {p}")
    return p


def read_mask_tif_aligned(path: str, shape, dst_transform, dst_crs_str, what: str, threshold: float = 0.5):
    p = require_file(path, what)
    H, W = shape
    dst_crs = CRS.from_string(dst_crs_str)

    with rasterio.open(p) as src:
        src_arr = src.read(1).astype(np.float32, copy=False)

        same = (
            src_arr.shape == (H, W)
            and (src.crs is not None)
            and (src.crs == dst_crs)
            and (src.transform == dst_transform)
        )

        if same:
            a = src_arr
        else:
            dst = np.zeros((H, W), dtype=np.float32)
            reproject(
                source=src_arr,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.max,
                src_nodata=src.nodata,
                dst_nodata=0.0,
                num_threads=1,
            )
            a = dst

    return (a > threshold).astype(np.uint8)


def dilate8(mask_u8: np.ndarray, iters: int = 1) -> np.ndarray:
    a = (mask_u8 > 0).astype(np.uint8)
    for _ in range(iters):
        p = np.pad(a, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        a = np.maximum.reduce(
            [
                p[0:-2, 0:-2], p[0:-2, 1:-1], p[0:-2, 2:],
                p[1:-1, 0:-2], p[1:-1, 1:-1], p[1:-1, 2:],
                p[2:, 0:-2], p[2:, 1:-1], p[2:, 2:],
            ]
        )
    return a.astype(np.uint8)


def check_raw_file_size(path: str, shape, dtype, what: str):
    p = require_file(path, what)
    H, W = shape
    expected = H * W * np.dtype(dtype).itemsize
    size = p.stat().st_size
    if size != expected:
        _die(
            f"[ERROR] size mismatch {what}: {p}\n"
            f"        bytes={size} expected={expected} "
            f"(H*W={H}*{W}, dtype={np.dtype(dtype).name})"
        )
    return p


def read_raw(path: str, shape, dtype, what: str, use_memmap: bool = False):
    check_raw_file_size(path, shape, dtype, what)
    H, W = shape
    if use_memmap:
        return np.memmap(path, mode="r", dtype=dtype, shape=(H, W))
    return np.fromfile(path, dtype=dtype, count=H * W).reshape((H, W))


def read_u8_tif(path: str, shape, what: str):
    p = require_file(path, what)
    with rasterio.open(p) as ds:
        a = ds.read(1)
    if a.dtype != np.uint8:
        a = a.astype(np.uint8, copy=False)
    if a.shape != shape:
        _die(
            f"[ERROR] shape mismatch {what}: {p}\n"
            f"        shape={a.shape} expected={shape}"
        )
    return a


def lat_band_t2(spec, H):
    a, b, c, d, e, f = spec["transform"]
    ys = np.arange(H, dtype=np.float32)
    lat = f + (ys + 0.5) * e
    abs_lat = np.abs(lat)

    t2 = np.zeros(H, dtype=np.uint8)
    t2[abs_lat < 23.5] = 3
    t2[(abs_lat >= 23.5) & (abs_lat < 45.0)] = 2
    t2[(abs_lat >= 45.0) & (abs_lat < 60.0)] = 1
    t2[abs_lat >= 60.0] = 0
    return t2


def map_worldcover_to_4bit(wc_dom_u8: np.ndarray, ocean_u8: np.ndarray):
    out = np.zeros_like(wc_dom_u8, dtype=np.uint8)

    out[wc_dom_u8 == 10] = 1
    out[wc_dom_u8 == 20] = 2
    out[wc_dom_u8 == 30] = 3
    out[wc_dom_u8 == 40] = 4
    out[wc_dom_u8 == 50] = 5
    out[wc_dom_u8 == 60] = 6
    out[wc_dom_u8 == 70] = 7
    out[wc_dom_u8 == 80] = 8
    out[wc_dom_u8 == 90] = 9
    out[wc_dom_u8 == 95] = 10
    out[wc_dom_u8 == 100] = 11

    out[ocean_u8 > 0] = 12
    return out


def compress_bin_to_2bit(x_u8: np.ndarray):
    y = x_u8.copy()
    y[y > 3] = 3
    return y


def qc(name: str, a: np.ndarray, show_topk: int = 8):
    arr = np.array(a, copy=False)
    u, c = np.unique(arr, return_counts=True)
    total = int(arr.size)
    top = sorted(zip(u.tolist(), c.tolist()), key=lambda x: x[1], reverse=True)[:show_topk]
    print(f"[QC] {name}: dtype={arr.dtype} shape={arr.shape} min={int(arr.min())} max={int(arr.max())} uniq={len(u)} total={total}")
    print(f"     top{show_topk}={top}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-spec", default="out/grid/grid_spec.json")

    ap.add_argument("--ocean-mask", default="out/ne/ne_ocean_mask.tif")
    ap.add_argument("--wc-dom", default="out/wc_full/wc_dom.u8")
    ap.add_argument("--tbin", default="out/worldclim/wc_tbin.u8")
    ap.add_argument("--pbin", default="out/worldclim/wc_pbin.u8")
    ap.add_argument("--ebin", default="out/worldclim/wc_ebin.u8")
    ap.add_argument("--area-u16", default="out/area/area_id.u16")
    ap.add_argument("--area-bits", type=int, default=10)

    ap.add_argument("--river-mask", default="out/ne/river_mask_merged.tif", help="river mask GeoTIFF (0/1)")
    ap.add_argument("--reef-mask", default="out/ne/ne_reef_mask.tif", help="reef mask GeoTIFF (0/1)")
    ap.add_argument("--river-dilate", type=int, default=1, help="dilate river mask by N iters (0=off)")

    ap.add_argument("--use-memmap", action="store_true")
    ap.add_argument("--out-bin", default="out/final/map32.bin")
    ap.add_argument("--out-legend", default="out/final/map32_legend.json")
    args = ap.parse_args()

    if not (1 <= args.area_bits <= 16):
        _die(f"[ERROR] area-bits must be 1..16: {args.area_bits}")

    spec, H, W = load_grid_spec(args.grid_spec)
    shape = (H, W)

    ocean = read_u8_tif(args.ocean_mask, shape, "ocean-mask")

    wc_dom = read_raw(args.wc_dom, shape, np.uint8, "wc-dom", use_memmap=args.use_memmap)
    tbin = read_raw(args.tbin, shape, np.uint8, "tbin", use_memmap=args.use_memmap)
    pbin = read_raw(args.pbin, shape, np.uint8, "pbin", use_memmap=args.use_memmap)
    ebin = read_raw(args.ebin, shape, np.uint8, "ebin", use_memmap=args.use_memmap)
    area = read_raw(args.area_u16, shape, np.uint16, "area-u16", use_memmap=args.use_memmap)

    dst_transform = Affine(*spec["transform"])
    dst_crs = spec["crs"]

    river_path = Path(args.river_mask)
    if river_path.exists():
        river = read_mask_tif_aligned(str(river_path), shape, dst_transform, dst_crs, "river-mask", threshold=0.5)
        if args.river_dilate > 0:
            river = dilate8(river, iters=args.river_dilate)
    else:
        print(f"[WARN] river-mask not found: {river_path} -> use zeros")
        river = np.zeros(shape, np.uint8)

    reef_path = Path(args.reef_mask)
    if reef_path.exists():
        reef = read_mask_tif_aligned(str(reef_path), shape, dst_transform, dst_crs, "reef-mask", threshold=0.5)
    else:
        print(f"[WARN] reef-mask not found: {reef_path} -> use zeros")
        reef = np.zeros(shape, np.uint8)

    qc("wc_dom", wc_dom)
    qc("tbin", tbin)
    qc("pbin", pbin)
    qc("ebin", ebin)
    qc("area", area)
    qc("river", river)
    qc("reef", reef)
    qc("ocean", ocean)

    wc_dom_np = np.array(wc_dom, dtype=np.uint8, copy=False)
    tbin_np = np.array(tbin, dtype=np.uint8, copy=False)
    pbin_np = np.array(pbin, dtype=np.uint8, copy=False)
    ebin_np = np.array(ebin, dtype=np.uint8, copy=False)
    area_np = np.array(area, dtype=np.uint16, copy=False)
    river_np = np.array(river, dtype=np.uint8, copy=False)
    reef_np = np.array(reef, dtype=np.uint8, copy=False)

    lc4 = map_worldcover_to_4bit(wc_dom_np, ocean)

    t2 = compress_bin_to_2bit(tbin_np)
    p2 = compress_bin_to_2bit(pbin_np)
    e2 = compress_bin_to_2bit(ebin_np)

    sea_rows_t2 = lat_band_t2(spec, H)
    sea = ocean > 0
    t2[sea] = sea_rows_t2[np.where(sea)[0]]
    e2[sea] = 0
    p2[sea] = 2

    water = ((wc_dom_np == 80) | (wc_dom_np == 90) | (river_np > 0) | (ocean > 0)).astype(np.uint8)
    built = (wc_dom_np == 50).astype(np.uint8)
    crop = (wc_dom_np == 40).astype(np.uint8)
    reef_f = (reef_np > 0).astype(np.uint8)
    forest = ((wc_dom_np == 10) | (wc_dom_np == 20) | (wc_dom_np == 95)).astype(np.uint8)
    ice = (wc_dom_np == 70).astype(np.uint8)
    flags6 = (water << 0) | (built << 1) | (crop << 2) | (reef_f << 3) | (forest << 4) | (ice << 5)

    def assert_range(name, a, lo, hi):
        mn = int(a.min())
        mx = int(a.max())
        if mn < lo or mx > hi:
            _die(f"[ERROR] range violation {name}: min={mn} max={mx} expected[{lo}..{hi}]")

    assert_range("lc4", lc4, 0, 15)
    assert_range("t2", t2, 0, 3)
    assert_range("e2", e2, 0, 3)
    assert_range("p2", p2, 0, 3)
    assert_range("flags6", flags6, 0, 63)
    assert_range("area_u16", area_np, 0, (1 << args.area_bits) - 1)

    b0 = (lc4 & 0x0F) | ((t2 & 0x03) << 4) | ((e2 & 0x03) << 6)
    b1 = ((p2 & 0x03) << 0) | ((flags6 & 0x3F) << 2)

    area_le = area_np.astype(np.uint16, copy=False)
    b2 = (area_le & 0x00FF).astype(np.uint8)
    b3 = ((area_le >> 8) & 0x00FF).astype(np.uint8)

    out = np.empty((H, W, 4), dtype=np.uint8)
    out[:, :, 0] = b0
    out[:, :, 1] = b1
    out[:, :, 2] = b2
    out[:, :, 3] = b3

    Path(args.out_bin).parent.mkdir(parents=True, exist_ok=True)
    out.reshape(-1).tofile(args.out_bin)

    legend = {
        "layout": {
            "byte0": "bits0-3 landcover4, bits4-5 t2, bits6-7 e2",
            "byte1": "bits0-1 p2, bits2-7 flags6(water,built,cropland,reef,forest,ice)",
            "byte2": "area10 low8",
            "byte3": "bits0-1 area10 high2, bits2-7 reserved6(0)",
        },
        "fields": {
            "landcover4": {"offset": 0, "bits": 4},
            "t2": {"offset": 4, "bits": 2},
            "e2": {"offset": 6, "bits": 2},
            "p2": {"offset": 8, "bits": 2},
            "flags6": {"offset": 10, "bits": 6},
            "area10": {"offset": 16, "bits": args.area_bits},
            "reserved6": {"offset": 16 + args.area_bits, "bits": 32 - (16 + args.area_bits)},
        },
        "compatibility": {
            "lo16_unchanged_from_map24": True,
            "reader_rule": "area_u16 = byte2 | (byte3 << 8); area10 = area_u16 & 0x03FF",
        },
        "landcover4": {
            "0": "UNKNOWN",
            "1": "TREE",
            "2": "SHRUB",
            "3": "GRASS",
            "4": "CROPLAND",
            "5": "BUILT",
            "6": "BARE",
            "7": "SNOW_ICE",
            "8": "WATER",
            "9": "WETLAND",
            "10": "MANGROVE",
            "11": "MOSS_LICHEN",
            "12": "OCEAN",
        },
        "t2": {"0": "POLAR", "1": "COLD", "2": "TEMPERATE", "3": "TROPICAL"},
        "e2": {"0": "LOW", "1": "MID", "2": "HIGH", "3": "ULTRA"},
        "p2": {"0": "ARID", "1": "LOW", "2": "MID", "3": "WET"},
        "flags6_bits": {
            "0": "water",
            "1": "built",
            "2": "cropland",
            "3": "reef",
            "4": "forest(tree+shrub+mangrove)",
            "5": "ice",
        },
        "area_bits": args.area_bits,
        "record_bytes": 4,
    }
    Path(args.out_legend).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_legend, "w", encoding="utf-8") as f:
        json.dump(legend, f, ensure_ascii=False, indent=2)

    out_size = Path(args.out_bin).stat().st_size
    expected_out = H * W * 4
    if out_size != expected_out:
        _die(f"[ERROR] out size mismatch: {args.out_bin} bytes={out_size} expected={expected_out}")

    print(f"[OK] wrote: {args.out_bin} ({out_size} bytes)")
    print(f"[OK] wrote: {args.out_legend}")


if __name__ == "__main__":
    main()
