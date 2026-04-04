#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import threading
from functools import lru_cache

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

W, H = 8640, 4320


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def latlon_to_xy(lat, lon):
    x = int(math.floor((lon + 180.0) / 360.0 * W))
    y = int(math.floor((90.0 - lat) / 180.0 * H))
    return clamp(x, 0, W - 1), clamp(y, 0, H - 1)


def xy_to_cell_bounds(x, y):
    dlon = 360.0 / W
    dlat = 180.0 / H
    lon_left = -180.0 + x * dlon
    lon_right = -180.0 + (x + 1) * dlon
    lat_top = 90.0 - y * dlat
    lat_bottom = 90.0 - (y + 1) * dlat
    return {
        "lat_top": lat_top,
        "lat_bottom": lat_bottom,
        "lon_left": lon_left,
        "lon_right": lon_right,
    }


def rect_bounds_from_wkt(wkt: str):
    if not wkt or not isinstance(wkt, str):
        return None

    m = re.search(r"POLYGON\s*\(\((.*)\)\)", wkt, flags=re.IGNORECASE)
    if not m:
        return None

    pts = []
    for part in m.group(1).split(","):
        part = part.strip()
        if not part:
            continue
        xy = part.split()
        if len(xy) < 2:
            continue
        lon = float(xy[0])
        lat = float(xy[1])
        pts.append((lon, lat))

    if not pts:
        return None

    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    return {
        "lon_left": min(lons),
        "lon_right": max(lons),
        "lat_bottom": min(lats),
        "lat_top": max(lats),
    }


@lru_cache(maxsize=8)
def load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Map32Reader:
    def __init__(self, path: str):
        self.f = open(path, "rb")
        self.lock = threading.Lock()

    def read_cell(self, idx: int):
        off = idx * 4
        with self.lock:
            self.f.seek(off)
            b = self.f.read(4)
        if len(b) != 4:
            raise ValueError("short read")
        return int.from_bytes(b, "little", signed=False)

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def decode_u32(u32: int):
    landcover4 = (u32 >> 0) & 0xF
    t2 = (u32 >> 4) & 0x3
    e2 = (u32 >> 6) & 0x3
    p2 = (u32 >> 8) & 0x3
    flags6 = (u32 >> 10) & 0x3F
    region10 = (u32 >> 16) & 0x3FF
    ext6_reserved = (u32 >> 26) & 0x3F
    return {
        "landcover4": landcover4,
        "t2": t2,
        "e2": e2,
        "p2": p2,
        "flags6": flags6,
        "flags_bits": [(flags6 >> i) & 1 for i in range(6)],
        "region10": region10,
        "ext6_reserved": ext6_reserved,
    }


def wc_name_from_legend(legend, code):
    if not isinstance(legend, dict):
        return None
    names = legend.get("landcover4")
    if isinstance(names, dict):
        return names.get(str(code)) or names.get(code)
    return None


def build_index_html():
    lines = [
        "<!doctype html>",
        '<html lang="ja">',
        "<head>",
        '  <meta charset="utf-8" />',
        "  <title>Map32 Viewer</title>",
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0" />',
        '  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="" />',
        "  <style>",
        "    body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }",
        "    #wrap { display:grid; grid-template-columns: 1fr 500px; height:100vh; }",
        "    #map { height:100%; }",
        "    #side { padding:12px; border-left:1px solid #ddd; overflow:auto; }",
        "    .row { display:flex; gap:8px; align-items:center; margin:8px 0; }",
        "    select { width:100%; padding:6px; }",
        "    .card { background:#f6f6f6; border-radius:10px; padding:10px; }",
        "    .k { color:#666; font-size:12px; margin-bottom:2px; }",
        "    .grid { display:grid; gap:10px; }",
        "    .muted { color:#666; font-size:12px; }",
        "    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }",
        "  </style>",
        "</head>",
        "<body>",
        '<div id="wrap">',
        '  <div id="map"></div>',
        '  <div id="side">',
        '    <h2 style="margin:0 0 6px;">Map32 確認ビューア</h2>',
        '    <div class="muted">地図クリック → map32.bin を4byte読んで、日本語一覧で表示します。</div>',
        '    <div class="row" style="margin-top:10px;">',
        '      <label style="min-width:90px;">オーバーレイ</label>',
        '      <select id="layer">',
        '        <option value="none">なし</option>',
        '        <option value="region">地域ID（濃淡）</option>',
        '        <option value="wc_dom">土地被覆（濃淡）</option>',
        '        <option value="flags">追加フラグ（濃淡）</option>',
        '        <option value="t2">気温4段階（濃淡）</option>',
        '        <option value="e2">標高4段階（濃淡）</option>',
        '        <option value="p2">降水4段階（濃淡）</option>',
        "      </select>",
        "    </div>",
        '    <div id="out" class="grid"><div class="card">クリック待ち…</div></div>',
        "  </div>",
        "</div>",
        '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>',
        "<script>",
        '  const map = L.map("map").setView([20, 0], 3);',
        '  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "&copy; OpenStreetMap" }).addTo(map);',
        "  let clickMarker = null;",
        "  let cellRect = null;",
        "  let regionRect = null;",
        "  let overlay = null;",
        '  const out = document.getElementById("out");',
        '  const sel = document.getElementById("layer");',
        '  const LANDCOVER_JA = {UNKNOWN:"不明",TREE:"森林",SHRUB:"低木",GRASS:"草地",CROPLAND:"農地",BUILT:"市街地",BARE:"裸地",SNOW_ICE:"雪氷",WATER:"内水域",WETLAND:"湿地",MANGROVE:"マングローブ",MOSS_LICHEN:"コケ・地衣類",OCEAN:"海"};',
        '  const T2_NAMES_JA = ["極寒冷", "寒冷", "温帯", "熱帯"];',
        '  const E2_NAMES_JA = ["低地", "中位", "高地", "超高地"];',
        '  const P2_NAMES_JA = ["極乾燥", "少なめ", "中間", "多雨"];',
        '  const FLAG_LABELS = ["水域", "市街地", "農地", "礁", "森林", "氷"];',
        "  function fmt(n, d=6){ try { return Number(n).toFixed(d); } catch { return String(n); } }",
        """  function escapeHtml(s){ return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;"); }""",
        "  function formatRegion(region10, meta){",
        "    if (!meta) return `地域ID ${region10}`;",
        "    const parts = [];",
        "    if (meta.name) parts.push(meta.name);",
        "    if (meta.name_ja) parts.push(meta.name_ja);",
        "    if (meta.country_name_ja) parts.push(`国:${meta.country_name_ja}`);",
        "    else if (meta.country_name_ja) parts.push(`Country:${meta.country_name_ja}`);",
        "    if (meta.country_iso_a3) parts.push(`ISO3:${meta.country_iso_a3}`);",
        "    parts.push(`ID:${region10}`);",
        '    return parts.join(" / ");',
        "  }",
        "  function renderInfo(j){",
        "    const d = j.decoded;",
        "    const landEn = d.wc_name ? String(d.wc_name) : null;",
        "    const landText = landEn ? (LANDCOVER_JA[landEn] || landEn) : `土地被覆 ${d.landcover4}`;",
        "    const rows = [",
        '      ["地域", formatRegion(d.region10, j.region_legend)],',
        '      ["土地", landText],',
        '      ["気温", T2_NAMES_JA[d.t2] ?? `気温=${d.t2}`],',
        '      ["標高", E2_NAMES_JA[d.e2] ?? `標高=${d.e2}`],',
        '      ["降水", P2_NAMES_JA[d.p2] ?? `降水=${d.p2}`],',
        '      [FLAG_LABELS[0], d.flags_bits[0] ? "有" : "無"],',
        '      [FLAG_LABELS[1], d.flags_bits[1] ? "有" : "無"],',
        '      [FLAG_LABELS[2], d.flags_bits[2] ? "有" : "無"],',
        '      [FLAG_LABELS[3], d.flags_bits[3] ? "有" : "無"],',
        '      [FLAG_LABELS[4], d.flags_bits[4] ? "有" : "無"],',
        '      [FLAG_LABELS[5], d.flags_bits[5] ? "有" : "無"],',
        "    ];",
        r"""    const lines = rows.map(([k,v]) => `${k}\t\t${v}`).join('\n');""",
        "    const b = j.cell.bounds;",
        """    const regionText = j.region_bounds ? `\nregion lat ${fmt(j.region_bounds.lat_bottom)}..${fmt(j.region_bounds.lat_top)} / lon ${fmt(j.region_bounds.lon_left)}..${fmt(j.region_bounds.lon_right)}` : ""; """,
        """    const meta = `\n\n---\nグリッド x=${j.x} y=${j.y} idx=${j.idx}\nlat ${fmt(b.lat_bottom)}..${fmt(b.lat_top)} / lon ${fmt(b.lon_left)}..${fmt(b.lon_right)}${regionText}\nraw u32=${j.raw.u32} / decode=${j.decode_layout} / ext6=${d.ext6_reserved}`;""",
        """    return `<div class="card"><div class="k">判定結果（日本語）</div><pre class="mono" style="margin:0; white-space:pre-wrap;">${escapeHtml(lines + meta)}</pre></div>`;""",
        "  }",
        "  function setOverlay(layerName){",
        "    if (overlay) { map.removeLayer(overlay); overlay = null; }",
        '    if (layerName === "none") return;',
        '    overlay = L.tileLayer(`/tile.png?z={z}&x={x}&y={y}&layer=${layerName}`, { maxZoom: 10, opacity: 0.55 });',
        "    overlay.addTo(map);",
        "  }",
        '  sel.addEventListener("change", () => setOverlay(sel.value));',
        "  setOverlay(sel.value);",
        '  map.on("click", async (e) => {',
        "    const { lat, lng } = e.latlng;",
        "    if (clickMarker) map.removeLayer(clickMarker);",
        "    clickMarker = L.circleMarker([lat, lng], { radius: 6 }).addTo(map);",
        "    try {",
        "      const r = await fetch(`/decode?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lng)}`);",
        "      const j = await r.json();",
        "      const b = j.cell.bounds;",
        "      if (cellRect) map.removeLayer(cellRect);",
        "      cellRect = L.rectangle([[b.lat_bottom, b.lon_left], [b.lat_top, b.lon_right]], { weight: 2 }).addTo(map);",
        "      if (regionRect) { map.removeLayer(regionRect); regionRect = null; }",
        "      if (j.region_bounds) {",
        "        const rb = j.region_bounds;",
        '        regionRect = L.rectangle([[rb.lat_bottom, rb.lon_left], [rb.lat_top, rb.lon_right]], { weight: 2, color: "#ff0000", fill: false }).addTo(map);',
        "      }",
        "      out.innerHTML = renderInfo(j);",
        "    } catch (err) {",
        """      out.innerHTML = `<div class="card">エラー: ${escapeHtml(String(err))}</div>`;""",
        "    }",
        "  });",
        "</script>",
        "</body>",
        "</html>",
    ]
    return "\n".join(lines)


def make_app(map_path, legend_path, area_legend_path, static_dir):
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    reader = Map32Reader(map_path)

    @app.on_event("shutdown")
    def _shutdown():
        reader.close()

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
            return f.read()

    @app.get("/decode")
    def decode(lat: float = Query(...), lon: float = Query(...)):
        x, y = latlon_to_xy(lat, lon)
        idx = y * W + x
        u32 = reader.read_cell(idx)
        dec = decode_u32(u32)

        legend = load_json(legend_path)
        area_legend = load_json(area_legend_path)

        dec["wc_name"] = wc_name_from_legend(legend, dec["landcover4"])

        region_meta = None
        if isinstance(area_legend, dict):
            id_to_meta = area_legend.get("id_to_meta")
            if isinstance(id_to_meta, dict):
                region_meta = id_to_meta.get(str(dec["region10"])) or id_to_meta.get(dec["region10"])

        region_bounds = None
        if isinstance(region_meta, dict):
            region_bounds = rect_bounds_from_wkt(region_meta.get("geometry_wkt"))

        cell_bounds = xy_to_cell_bounds(x, y)

        return {
            "lat": lat,
            "lon": lon,
            "x": x,
            "y": y,
            "idx": idx,
            "cell": {"bounds": cell_bounds},
            "raw": {"u32": u32},
            "decoded": dec,
            "region_legend": region_meta,
            "region_bounds": region_bounds,
            "decode_layout": "map32 fixed layout",
        }

    @app.get("/tile.png")
    def tile_png(z: int, x: int, y: int, layer: str = "region"):
        from PIL import Image

        tile = Image.new("RGB", (256, 256))
        px = tile.load()

        def tile2lon(tx, tz):
            return tx / (2**tz) * 360.0 - 180.0

        def tile2lat(ty, tz):
            n = math.pi - 2.0 * math.pi * ty / (2**tz)
            return math.degrees(math.atan(math.sinh(n)))

        lon0 = tile2lon(x, z)
        lon1 = tile2lon(x + 1, z)
        lat0 = tile2lat(y, z)
        lat1 = tile2lat(y + 1, z)

        for j in range(256):
            lat = lat0 + (lat1 - lat0) * (j / 255.0)
            for i in range(256):
                lon = lon0 + (lon1 - lon0) * (i / 255.0)
                gx, gy = latlon_to_xy(lat, lon)
                idx = gy * W + gx
                u32 = reader.read_cell(idx)
                d = decode_u32(u32)

                if layer == "region":
                    v = (d["region10"] * 37) % 256
                elif layer == "wc_dom":
                    v = d["landcover4"] * 17
                elif layer == "flags":
                    v = d["flags6"] * 4
                elif layer == "t2":
                    v = d["t2"] * 85
                elif layer == "e2":
                    v = d["e2"] * 85
                elif layer == "p2":
                    v = d["p2"] * 85
                else:
                    v = 0

                px[i, j] = (v, v, v)

        import io

        buf = io.BytesIO()
        tile.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map32", default="map32.bin")
    ap.add_argument("--legend", default="map32_legend.json")
    ap.add_argument("--area-legend", default="country_area_legend_patched.json")
    ap.add_argument("--static-dir", default="static_map32")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    os.makedirs(args.static_dir, exist_ok=True)

    with open(os.path.join(args.static_dir, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(build_index_html())

    app = make_app(args.map32, args.legend, args.area_legend, args.static_dir)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()