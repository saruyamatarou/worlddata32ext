#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_grid_spec(path: str):
    spec = load_json(path)
    H = int(spec["height"])
    W = int(spec["width"])
    a, b, c, d, e, f = spec["transform"]
    if abs(b) > 1e-12 or abs(d) > 1e-12:
        raise SystemExit("grid_spec の transform は回転なし前提です")
    if a <= 0:
        raise SystemExit("grid_spec の transform[0] は正である必要があります")
    if e >= 0:
        raise SystemExit("grid_spec の transform[4] は北->南の負値である必要があります")
    return spec, H, W, float(a), float(c), float(e), float(f)


class Map32Dataset:
    def __init__(
        self,
        grid_spec_path: str,
        map32_bin_path: str,
        map32_legend_path: Optional[str] = None,
        area_legend_path: Optional[str] = None,
        country_area_path: Optional[str] = None,
    ):
        self.grid_spec, self.H, self.W, self.dx, self.x0, self.dy, self.y0 = load_grid_spec(grid_spec_path)
        self.record_bytes = 4
        self.map32_bin_path = Path(map32_bin_path)
        if not self.map32_bin_path.exists():
            raise SystemExit(f"missing map32.bin: {self.map32_bin_path}")
        expected = self.H * self.W * self.record_bytes
        actual = self.map32_bin_path.stat().st_size
        if actual != expected:
            raise SystemExit(
                f"map32.bin size mismatch: {self.map32_bin_path} bytes={actual} expected={expected}"
            )

        self.raw = np.memmap(self.map32_bin_path, mode="r", dtype=np.uint8)

        self.legend = load_json(map32_legend_path) if map32_legend_path and Path(map32_legend_path).exists() else {}
        self.record_bytes = int(self.legend.get("record_bytes", 4))
        self.area_bits = int(self.legend.get("area_bits", 10))
        self.area_mask = (1 << self.area_bits) - 1 if self.area_bits < 16 else 0xFFFF

        self.landcover_names = self.legend.get("landcover4", {
            "0": "UNKNOWN", "1": "TREE", "2": "SHRUB", "3": "GRASS", "4": "CROPLAND",
            "5": "BUILT", "6": "BARE", "7": "SNOW_ICE", "8": "WATER", "9": "WETLAND",
            "10": "MANGROVE", "11": "MOSS_LICHEN", "12": "OCEAN",
        })
        self.t2_names = self.legend.get("t2", {"0": "POLAR", "1": "COLD", "2": "TEMPERATE", "3": "TROPICAL"})
        self.e2_names = self.legend.get("e2", {"0": "LOW", "1": "MID", "2": "HIGH", "3": "ULTRA"})
        self.p2_names = self.legend.get("p2", {"0": "ARID", "1": "LOW", "2": "MID", "3": "WET"})
        self.flag_names = self.legend.get("flags6_bits", {
            "0": "water", "1": "built", "2": "cropland", "3": "reef", "4": "forest", "5": "ice"
        })

        self.area_legend = load_json(area_legend_path) if area_legend_path and Path(area_legend_path).exists() else {}
        self.area_meta = self.area_legend.get("id_to_meta", {})
        self.country_area_rows = {}
        if country_area_path and Path(country_area_path).exists():
            try:
                rows = load_json(country_area_path)
                if isinstance(rows, list):
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        aid = row.get("area_id")
                        if aid is None:
                            continue
                        self.country_area_rows[str(aid)] = row
            except Exception:
                pass

    def lonlat_to_xy(self, lon: float, lat: float):
        lon = min(max(lon, -180.0), math.nextafter(180.0, -math.inf))
        lat = min(max(lat, -90.0), 90.0)

        x = int(math.floor((lon - self.x0) / self.dx))
        y = int(math.floor((lat - self.y0) / self.dy))

        x = min(max(x, 0), self.W - 1)
        y = min(max(y, 0), self.H - 1)
        return x, y

    def xy_to_lonlat_center(self, x: int, y: int):
        lon = self.x0 + (x + 0.5) * self.dx
        lat = self.y0 + (y + 0.5) * self.dy
        return lon, lat

    def cell_bbox(self, x: int, y: int):
        left = self.x0 + x * self.dx
        right = self.x0 + (x + 1) * self.dx
        top = self.y0 + y * self.dy
        bottom = self.y0 + (y + 1) * self.dy
        return {
            "lon_left": min(left, right),
            "lon_right": max(left, right),
            "lat_bottom": min(bottom, top),
            "lat_top": max(bottom, top),
        }

    def area_rect_geojson(self, area_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        wkt = str(area_row.get("geometry_wkt", "") or "")
        if not wkt:
            return None
        import re
        nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", wkt)]
        if len(nums) < 8 or len(nums) % 2 != 0:
            return None
        pts = [[nums[i], nums[i+1]] for i in range(0, len(nums), 2)]
        return {"type": "Polygon", "coordinates": [pts]}

    def read_record(self, x: int, y: int):
        if not (0 <= x < self.W and 0 <= y < self.H):
            raise IndexError((x, y))
        base = (y * self.W + x) * self.record_bytes
        b0 = int(self.raw[base + 0])
        b1 = int(self.raw[base + 1])
        b2 = int(self.raw[base + 2])
        b3 = int(self.raw[base + 3])
        lo16 = b0 | (b1 << 8)
        area_u16 = b2 | (b3 << 8)
        area_id = area_u16 & self.area_mask
        return b0, b1, b2, b3, lo16, area_u16, area_id

    def decode_xy(self, x: int, y: int) -> Dict[str, Any]:
        b0, b1, b2, b3, lo16, area_u16, area_id = self.read_record(x, y)
        lc4 = b0 & 0x0F
        t2 = (b0 >> 4) & 0x03
        e2 = (b0 >> 6) & 0x03
        p2 = b1 & 0x03
        flags6 = (b1 >> 2) & 0x3F
        flag_bits = [(flags6 >> i) & 1 for i in range(6)]
        flag_labels_on = [self.flag_names.get(str(i), f"bit{i}") for i, v in enumerate(flag_bits) if v]

        lon_center, lat_center = self.xy_to_lonlat_center(x, y)
        bbox = self.cell_bbox(x, y)
        area_meta = self.area_meta.get(str(area_id))
        area_row = self.country_area_rows.get(str(area_id))
        area_rect = self.area_rect_geojson(area_row) if area_row else None

        return {
            "grid": {
                "width": self.W,
                "height": self.H,
                "x": x,
                "y": y,
                "cell_id": y * self.W + x,
                "center_lon": lon_center,
                "center_lat": lat_center,
                "bbox": bbox,
            },
            "bytes": {
                "b0": b0,
                "b1": b1,
                "b2": b2,
                "b3": b3,
                "lo16": lo16,
                "area_u16": area_u16,
            },
            "decoded": {
                "landcover4": lc4,
                "landcover4_name": self.landcover_names.get(str(lc4), "UNKNOWN"),
                "t2": t2,
                "t2_name": self.t2_names.get(str(t2), "UNKNOWN"),
                "e2": e2,
                "e2_name": self.e2_names.get(str(e2), "UNKNOWN"),
                "p2": p2,
                "p2_name": self.p2_names.get(str(p2), "UNKNOWN"),
                "flags6": flags6,
                "flags_bits": flag_bits,
                "flags_on": flag_labels_on,
                "area_id": area_id,
                "reserved_high_bits": area_u16 >> self.area_bits,
            },
            "area_meta": area_meta,
            "area_row": area_row,
            "cell_geojson": {
                "type": "Polygon",
                "coordinates": [[
                    [bbox["lon_left"], bbox["lat_top"]],
                    [bbox["lon_right"], bbox["lat_top"]],
                    [bbox["lon_right"], bbox["lat_bottom"]],
                    [bbox["lon_left"], bbox["lat_bottom"]],
                    [bbox["lon_left"], bbox["lat_top"]],
                ]],
            },
            "area_rect_geojson": area_rect,
        }



def build_index_html() -> str:
    return """<!doctype html>
<html lang=\"ja\">
<head>
  <meta charset=\"utf-8\" />
  <title>map32 click viewer</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" crossorigin=\"\" />
  <style>
    html, body { margin: 0; height: 100%; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
    #wrap { display: grid; grid-template-columns: 1fr 420px; height: 100%; }
    #map { height: 100%; }
    #side { border-left: 1px solid #ddd; padding: 12px; overflow: auto; }
    .row { display: flex; gap: 8px; margin: 8px 0; }
    input, button { padding: 8px; font-size: 14px; }
    input { width: 100%; }
    .card { background: #f6f6f6; border-radius: 10px; padding: 10px; margin-top: 10px; }
    .muted { color: #666; font-size: 12px; }
    .mono { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
  </style>
</head>
<body>
<div id=\"wrap\">
  <div id=\"map\"></div>
  <div id=\"side\">
    <h2 style=\"margin:0 0 6px;\">map32 クリックビューア</h2>
    <div class=\"muted\">地図をクリックすると、そのセルの map32 デコード結果・area_id・エリア名を表示します。</div>
    <div class=\"row\">
      <input id=\"lat\" placeholder=\"lat\" value=\"35.6812\" />
      <input id=\"lon\" placeholder=\"lon\" value=\"139.7671\" />
      <button id=\"goLatLon\">移動</button>
    </div>
    <div class=\"row\">
      <input id=\"x\" placeholder=\"x\" />
      <input id=\"y\" placeholder=\"y\" />
      <button id=\"goXY\">XY</button>
    </div>
    <div id=\"summary\" class=\"card\">未選択</div>
    <div id=\"json\" class=\"card mono\">未選択</div>
  </div>
</div>
<script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\" crossorigin=\"\"></script>
<script>
const map = L.map('map').setView([20, 0], 2);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap'
}).addTo(map);
let marker = null;
let cellLayer = null;
let areaLayer = null;
const summary = document.getElementById('summary');
const jsonBox = document.getElementById('json');
const latInput = document.getElementById('lat');
const lonInput = document.getElementById('lon');
const xInput = document.getElementById('x');
const yInput = document.getElementById('y');

actionSummary('地図をクリックしてください。');

function esc(s){
  return String(s)
    .replaceAll('&','&amp;')
    .replaceAll('<','&lt;')
    .replaceAll('>','&gt;')
    .replaceAll('"','&quot;')
    .replaceAll("'", '&#039;');
}

function actionSummary(html){ summary.innerHTML = html; }

function clearOverlays(){
  if (cellLayer) { map.removeLayer(cellLayer); cellLayer = null; }
  if (areaLayer) { map.removeLayer(areaLayer); areaLayer = null; }
}

function renderInfo(j){
  const d = j.decoded || {};
  const g = j.grid || {};
  const a = j.area_meta || {};
  const lines = [];
  lines.push(`xy=(${g.x}, ${g.y}) cellId=${g.cell_id}`);
  lines.push(`landcover=${d.landcover4_name} (${d.landcover4})`);
  lines.push(`temp=${d.t2_name} (${d.t2}) / elev=${d.e2_name} (${d.e2}) / precip=${d.p2_name} (${d.p2})`);
  lines.push(`flags=${(d.flags_on || []).join(', ') || '(none)'}`);
  lines.push(`area_id=${d.area_id}`);
  if (a && Object.keys(a).length) {
    lines.push(`area=${a.name || ''} / ${a.name_en || ''}`);
    lines.push(`source_id=${a.source_id || ''} / iso=${a.iso_a3 || ''}`);
    if (a.country_name_ja || a.country_name_en) {
      lines.push(`country=${a.country_name_ja || ''} / ${a.country_name_en || ''}`);
    }
  }
  actionSummary(`<div class=\"mono\">${esc(lines.join('\\n'))}</div>`);
  jsonBox.textContent = JSON.stringify(j, null, 2);
}

async function decodeByLatLon(lat, lon){
  const r = await fetch(`/decode?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}`);
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

async function decodeByXY(x, y){
  const r = await fetch(`/decode_xy?x=${encodeURIComponent(x)}&y=${encodeURIComponent(y)}`);
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

function drawResult(j){
  clearOverlays();
  const center = [j.grid.center_lat, j.grid.center_lon];
  if (marker) map.removeLayer(marker);
  marker = L.marker(center).addTo(map);
  marker.bindPopup(`x=${j.grid.x}, y=${j.grid.y}<br>area=${j.decoded.area_id}`).openPopup();
  cellLayer = L.geoJSON(j.cell_geojson, { style: { color: '#0070f3', weight: 2, fill: false } }).addTo(map);
  if (j.area_rect_geojson) {
    areaLayer = L.geoJSON(j.area_rect_geojson, { style: { color: '#ff0000', weight: 2, fill: false } }).addTo(map);
  }
  renderInfo(j);
}

async function jumpLatLon(){
  const lat = Number(latInput.value);
  const lon = Number(lonInput.value);
  const j = await decodeByLatLon(lat, lon);
  xInput.value = j.grid.x;
  yInput.value = j.grid.y;
  drawResult(j);
  map.panTo([j.grid.center_lat, j.grid.center_lon]);
}

async function jumpXY(){
  const x = Number(xInput.value);
  const y = Number(yInput.value);
  const j = await decodeByXY(x, y);
  latInput.value = j.grid.center_lat.toFixed(6);
  lonInput.value = j.grid.center_lon.toFixed(6);
  drawResult(j);
  map.panTo([j.grid.center_lat, j.grid.center_lon]);
}

document.getElementById('goLatLon').addEventListener('click', async () => {
  try { await jumpLatLon(); }
  catch (e) { actionSummary(`<div class=\"mono\">${esc(String(e))}</div>`); }
});

document.getElementById('goXY').addEventListener('click', async () => {
  try { await jumpXY(); }
  catch (e) { actionSummary(`<div class=\"mono\">${esc(String(e))}</div>`); }
});

map.on('click', async (ev) => {
  latInput.value = ev.latlng.lat.toFixed(6);
  lonInput.value = ev.latlng.lng.toFixed(6);
  try {
    const j = await decodeByLatLon(ev.latlng.lat, ev.latlng.lng);
    xInput.value = j.grid.x;
    yInput.value = j.grid.y;
    drawResult(j);
  } catch (e) {
    actionSummary(`<div class=\"mono\">${esc(String(e))}</div>`);
  }
});

(async () => {
  try {
    const r = await fetch('/meta');
    const j = await r.json();
    actionSummary(`<div class=\"mono\">grid=${j.width}x${j.height}\nrecord_bytes=${j.record_bytes}\narea_bits=${j.area_bits}\n地図をクリックしてください。</div>`);
  } catch (e) {
    actionSummary(`<div class=\"mono\">${esc(String(e))}</div>`);
  }
})();
</script>
</body>
</html>
"""


def build_app(dataset: Map32Dataset, static_dir: str) -> FastAPI:
    app = FastAPI()
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
            return f.read()

    @app.get("/meta")
    def meta():
        return {
            "width": dataset.W,
            "height": dataset.H,
            "record_bytes": dataset.record_bytes,
            "area_bits": dataset.area_bits,
            "grid_spec": {
                "transform": dataset.grid_spec["transform"],
                "crs": dataset.grid_spec.get("crs", "EPSG:4326"),
            },
            "paths": {
                "map32_bin": str(dataset.map32_bin_path),
            },
        }

    @app.get("/decode")
    def decode(lat: float = Query(...), lon: float = Query(...)):
        x, y = dataset.lonlat_to_xy(lon, lat)
        return dataset.decode_xy(x, y)

    @app.get("/decode_xy")
    def decode_xy(x: int = Query(...), y: int = Query(...)):
        if not (0 <= x < dataset.W and 0 <= y < dataset.H):
            raise HTTPException(status_code=400, detail=f"x/y out of range: {(x, y)}")
        return dataset.decode_xy(x, y)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-spec", default="out/grid/grid_spec.json")
    ap.add_argument("--map32-bin", default="out/final/map32.bin")
    ap.add_argument("--map32-legend", default="out/final/map32_legend.json")
    ap.add_argument("--area-legend", default="out/area/area_legend.json")
    ap.add_argument("--country-area", default="country_area_with_area_id.json")
    ap.add_argument("--static-dir", default="static_map32_click_viewer")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()

    os.makedirs(args.static_dir, exist_ok=True)
    with open(os.path.join(args.static_dir, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(build_index_html())

    dataset = Map32Dataset(
        grid_spec_path=args.grid_spec,
        map32_bin_path=args.map32_bin,
        map32_legend_path=args.map32_legend,
        area_legend_path=args.area_legend,
        country_area_path=args.country_area,
    )
    app = build_app(dataset, args.static_dir)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
