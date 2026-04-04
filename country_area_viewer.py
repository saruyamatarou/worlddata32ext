#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import shapefile
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

AREA_ID_RE = re.compile(r"^([A-Za-z0-9_]+)-\d+$")


def norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def parse_wkt_rect(wkt: str) -> Optional[Dict[str, float]]:
    if not wkt or not isinstance(wkt, str):
        return None
    m = re.search(r"POLYGON\s*\(\((.*)\)\)", wkt, flags=re.IGNORECASE)
    if not m:
        return None

    pts: List[Tuple[float, float]] = []
    for part in m.group(1).split(","):
        xy = part.strip().split()
        if len(xy) < 2:
            continue
        try:
            lon = float(xy[0])
            lat = float(xy[1])
        except ValueError:
            continue
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


def bbox_intersects(a: Dict[str, float], b: Dict[str, float]) -> bool:
    return not (
        a["lon_right"] < b["lon_left"]
        or a["lon_left"] > b["lon_right"]
        or a["lat_top"] < b["lat_bottom"]
        or a["lat_bottom"] > b["lat_top"]
    )


def load_country_area_any(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        pass

    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    out: List[Dict[str, Any]] = []

    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, j = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue

        if isinstance(obj, list):
            out.extend([x for x in obj if isinstance(x, dict)])
        elif isinstance(obj, dict):
            out.append(obj)

        i = j

    return out


def shp_points_to_geojson_rings(shape: shapefile.Shape) -> List[List[List[float]]]:
    points = shape.points
    parts = list(shape.parts) + [len(points)]
    rings: List[List[List[float]]] = []

    for i in range(len(parts) - 1):
        ring_pts = points[parts[i] : parts[i + 1]]
        rings.append([[float(x), float(y)] for x, y in ring_pts])

    return rings


def shape_to_geojson_geometry(shape: shapefile.Shape) -> Dict[str, Any]:
    rings = shp_points_to_geojson_rings(shape)
    if len(rings) == 1:
        return {"type": "Polygon", "coordinates": [rings[0]]}
    return {"type": "MultiPolygon", "coordinates": [[r] for r in rings]}


class CountryIndex:
    def __init__(self, shp_path: str):
        self.reader = shapefile.Reader(shp_path, encoding="utf-8")
        self.fields = [f[0] for f in self.reader.fields[1:]]
        self.records = self.reader.records()
        self.shapes = self.reader.shapes()

        self.items: List[Dict[str, Any]] = []
        for rec, shp in zip(self.records, self.shapes):
            props = {k: rec[i] for i, k in enumerate(self.fields)}
            bbox = {
                "lon_left": float(shp.bbox[0]),
                "lat_bottom": float(shp.bbox[1]),
                "lon_right": float(shp.bbox[2]),
                "lat_top": float(shp.bbox[3]),
            }
            self.items.append(
                {
                    "props": props,
                    "shape": shp,
                    "bbox": bbox,
                    "aliases": self._aliases(props),
                }
            )

    def _aliases(self, p: Dict[str, Any]) -> set:
        keys = set()
        for k in [
            "ADM0_A3",
            "SOV_A3",
            "GU_A3",
            "ISO_A2",
            "ISO_A3_EH",
            "ISO_A3",
            "ADMIN",
            "NAME",
            "NAME_EN",
            "NAME_LONG",
            "SOVEREIGNT",
            "BRK_NAME",
        ]:
            v = p.get(k)
            if v not in (None, "", "-99"):
                keys.add(norm(v))
        return keys

    def list_countries(self) -> List[Dict[str, str]]:
        out = []
        seen = set()
        for item in self.items:
            p = item["props"]
            code = str(p.get("ADM0_A3") or p.get("SOV_A3") or p.get("ISO_A3") or "")
            name = str(p.get("ADMIN") or p.get("NAME") or p.get("NAME_LONG") or code)
            key = (code, name)
            if key in seen:
                continue
            seen.add(key)
            out.append({"code": code, "name": name})
        out.sort(key=lambda x: (x["name"], x["code"]))
        return out

    def find_one(self, key: str) -> Dict[str, Any]:
        q = norm(key)

        for item in self.items:
            if q in item["aliases"]:
                return item

        for item in self.items:
            if any(q in a for a in item["aliases"] if a):
                return item

        raise KeyError(key)


def build_app(ne_shp: str, country_area_path: str, static_dir: str) -> FastAPI:
    app = FastAPI()
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    area_rows = load_country_area_any(country_area_path)
    country_idx = CountryIndex(ne_shp)

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
            return f.read()

    @app.get("/countries")
    def countries():
        return {"countries": country_idx.list_countries()}

    @app.get("/country_view")
    def country_view(key: str = Query(..., description="ISO3 or country name")):
        try:
            item = country_idx.find_one(key)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"country not found: {key}")

        props = item["props"]
        shp = item["shape"]
        country_bbox = item["bbox"]

        country_geom = shape_to_geojson_geometry(shp)
        country_feature = {
            "type": "Feature",
            "properties": {
                "code": str(props.get("ADM0_A3") or props.get("SOV_A3") or props.get("ISO_A3") or ""),
                "name": str(props.get("ADMIN") or props.get("NAME") or props.get("NAME_LONG") or ""),
            },
            "geometry": country_geom,
        }

        code_candidates = set()
        for k in ["ADM0_A3", "SOV_A3", "ISO_A2", "ISO_A3", "GU_A3"]:
            v = props.get(k)
            if v not in (None, "", "-99"):
                code_candidates.add(str(v).upper())
        code_candidates |= {c[:2] for c in list(code_candidates) if len(c) >= 2}

        area_features = []
        for row in area_rows:
            area_id = str(row.get("id", ""))
            m = AREA_ID_RE.match(area_id)
            prefix = m.group(1).upper() if m else ""

            rect = parse_wkt_rect(row.get("geometry_wkt", ""))
            if rect is None:
                continue

            matched = prefix in code_candidates

            if not matched and bbox_intersects(rect, country_bbox):
                name_blob = " ".join(
                    [
                        str(row.get("id", "")),
                        str(row.get("name", "")),
                        str(row.get("name_en", "")),
                        str(row.get("reason", "")),
                    ]
                ).lower()
                country_name = str(props.get("ADMIN") or props.get("NAME") or "").lower()
                if country_name and country_name in name_blob:
                    matched = True

            if not matched:
                continue

            poly = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [rect["lon_left"], rect["lat_top"]],
                        [rect["lon_right"], rect["lat_top"]],
                        [rect["lon_right"], rect["lat_bottom"]],
                        [rect["lon_left"], rect["lat_bottom"]],
                        [rect["lon_left"], rect["lat_top"]],
                    ]
                ],
            }

            area_features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "id": area_id,
                        "name": row.get("name", ""),
                        "name_en": row.get("name_en", ""),
                        "reason": row.get("reason", ""),
                        "prefix": prefix,
                    },
                    "geometry": poly,
                }
            )

        return {
            "country": country_feature,
            "areas": {"type": "FeatureCollection", "features": area_features},
            "bbox": country_bbox,
            "summary": {
                "code": country_feature["properties"]["code"],
                "name": country_feature["properties"]["name"],
                "area_count": len(area_features),
            },
        }

    return app


def build_index_html() -> str:
    lines = [
        "<!doctype html>",
        '<html lang="ja">',
        "<head>",
        '  <meta charset="utf-8" />',
        "  <title>Country Area Viewer</title>",
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0" />',
        '  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="" />',
        "  <style>",
        "    body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }",
        "    #wrap { display:grid; grid-template-columns: 1fr 420px; height:100vh; }",
        "    #map { height:100%; }",
        "    #side { padding:12px; border-left:1px solid #ddd; overflow:auto; }",
        "    .row { display:flex; gap:8px; align-items:center; margin:8px 0; }",
        "    input, button, select { padding:8px; font-size:14px; }",
        "    input { flex:1; }",
        "    .card { background:#f6f6f6; border-radius:10px; padding:10px; margin-top:10px; }",
        "    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; white-space:pre-wrap; }",
        "    .muted { color:#666; font-size:12px; }",
        "  </style>",
        "</head>",
        "<body>",
        '<div id="wrap">',
        '  <div id="map"></div>',
        '  <div id="side">',
        '    <h2 style="margin:0 0 6px;">国境 + country_area 確認ビューア</h2>',
        '    <div class="muted">ISO3 や国名を入れると、Natural Earth の国境と country_area.json の分割矩形を重ねて表示します。</div>',
        '    <div class="row">',
        '      <input id="key" value="JPN" placeholder="例: JPN / Japan / 日本" />',
        '      <button id="loadBtn">表示</button>',
        "    </div>",
        '    <div class="row">',
        '      <select id="countrySelect"><option value="">国一覧を読み込み中...</option></select>',
        "    </div>",
        '    <div id="out" class="card">未表示</div>',
        "  </div>",
        "</div>",
        '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>',
        "<script>",
        "  const map = L.map('map').setView([20, 0], 2);",
        "  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap' }).addTo(map);",
        "  const out = document.getElementById('out');",
        "  const keyInput = document.getElementById('key');",
        "  const loadBtn = document.getElementById('loadBtn');",
        "  const countrySelect = document.getElementById('countrySelect');",
        "  let countryLayer = null;",
        "  let areasLayer = null;",
        "  function escapeHtml(s){",
        """    return String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#039;');""",
        "  }",
        "  function renderSummary(j){",
        "    const rows = [];",
        "    rows.push(`国: ${j.summary.name} (${j.summary.code})`);",
        "    rows.push(`矩形数: ${j.summary.area_count}`);",
        "    rows.push('---');",
        "    for (const f of j.areas.features) {",
        "      rows.push(`${f.properties.id} / ${f.properties.name || ''} / ${f.properties.name_en || ''}`);",
        "    }",
        """    out.innerHTML = `<div class="mono">${escapeHtml(rows.join('\\n'))}</div>`;""",
        "  }",
        "  function clearLayers(){",
        "    if (countryLayer) { map.removeLayer(countryLayer); countryLayer = null; }",
        "    if (areasLayer) { map.removeLayer(areasLayer); areasLayer = null; }",
        "  }",
        "  async function loadCountries(){",
        "    const r = await fetch('/countries');",
        "    const j = await r.json();",
        """    countrySelect.innerHTML = '<option value="">国一覧から選択</option>';""",
        "    for (const c of j.countries) {",
        "      const opt = document.createElement('option');",
        "      opt.value = c.code;",
        "      opt.textContent = `${c.name} (${c.code})`;",
        "      countrySelect.appendChild(opt);",
        "    }",
        "  }",
        "  async function loadCountry(key){",
        "    const r = await fetch(`/country_view?key=${encodeURIComponent(key)}`);",
        "    if (!r.ok) throw new Error(await r.text());",
        "    const j = await r.json();",
        "    clearLayers();",
        "    countryLayer = L.geoJSON(j.country, { style: { color: '#0070f3', weight: 3, fill: false } }).addTo(map);",
        "    areasLayer = L.geoJSON(j.areas, {",
        "      style: { color: '#ff0000', weight: 2, fill: false },",
        "      onEachFeature: function(feature, layer){",
        "        const p = feature.properties || {};",
        "        layer.bindPopup(",
        """          `<b>${escapeHtml(p.id || '')}</b><br>${escapeHtml(p.name || '')}<br>${escapeHtml(p.name_en || '')}<br><div class="mono">${escapeHtml(p.reason || '')}</div>`""",
        "        );",
        "      }",
        "    }).addTo(map);",
        "    const group = L.featureGroup([countryLayer, areasLayer]);",
        "    map.fitBounds(group.getBounds(), { padding: [20, 20] });",
        "    renderSummary(j);",
        "  }",
        "  loadBtn.addEventListener('click', async () => {",
        "    try { await loadCountry(keyInput.value.trim()); }",
        """    catch (e) { out.innerHTML = `<div class="mono">${escapeHtml(String(e))}</div>`; }""",
        "  });",
        "  countrySelect.addEventListener('change', async () => {",
        "    if (!countrySelect.value) return;",
        "    keyInput.value = countrySelect.value;",
        "    try { await loadCountry(countrySelect.value); }",
        """    catch (e) { out.innerHTML = `<div class="mono">${escapeHtml(String(e))}</div>`; }""",
        "  });",
        "  (async () => {",
        "    try {",
        "      await loadCountries();",
        "      await loadCountry(keyInput.value.trim());",
        "    } catch (e) {",
        """      out.innerHTML = `<div class="mono">${escapeHtml(String(e))}</div>`;""",
        "    }",
        "  })();",
        "</script>",
        "</body>",
        "</html>",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ne-shp", required=True, help="ne_10m_admin_0_countries.shp")
    ap.add_argument("--country-area", required=True, help="country_area.json")
    ap.add_argument("--static-dir", default="static_country_area_viewer")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()

    os.makedirs(args.static_dir, exist_ok=True)
    with open(os.path.join(args.static_dir, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(build_index_html())

    app = build_app(args.ne_shp, args.country_area, args.static_dir)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()