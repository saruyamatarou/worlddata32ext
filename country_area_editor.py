#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime as dt
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

import shapefile
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

AREA_ID_RE = re.compile(r'^([A-Za-z0-9_]+)-\d+$')


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
            lon = float(xy[0]); lat = float(xy[1])
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


def rect_to_wkt(lon_left: float, lon_right: float, lat_bottom: float, lat_top: float) -> str:
    return (
        f"POLYGON (({lon_left:.6f} {lat_top:.6f}, "
        f"{lon_right:.6f} {lat_top:.6f}, "
        f"{lon_right:.6f} {lat_bottom:.6f}, "
        f"{lon_left:.6f} {lat_bottom:.6f}, "
        f"{lon_left:.6f} {lat_top:.6f}))"
    )


def bbox_intersects(a: Dict[str, float], b: Dict[str, float]) -> bool:
    return not (
        a["lon_right"] < b["lon_left"] or
        a["lon_left"] > b["lon_right"] or
        a["lat_top"] < b["lat_bottom"] or
        a["lat_bottom"] > b["lat_top"]
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


def save_country_area(path: str, rows: List[Dict[str, Any]]) -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak_path = f"{path}.{ts}.bak"
    if os.path.exists(path):
        shutil.copy2(path, bak_path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return bak_path


def shp_points_to_geojson_rings(shape: shapefile.Shape) -> List[List[List[float]]]:
    points = shape.points
    parts = list(shape.parts) + [len(points)]
    rings: List[List[List[float]]] = []
    for i in range(len(parts) - 1):
        ring_pts = points[parts[i]:parts[i + 1]]
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
            self.items.append({
                "props": props,
                "shape": shp,
                "bbox": bbox,
                "aliases": self._aliases(props),
            })

    def _aliases(self, p: Dict[str, Any]) -> set:
        keys = set()
        for k in ["ADM0_A3","SOV_A3","GU_A3","ISO_A2","ISO_A3_EH","ISO_A3","ADMIN","NAME","NAME_EN","NAME_LONG","SOVEREIGNT","BRK_NAME"]:
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


class AreaUpdateRequest(BaseModel):
    id: str
    lon_left: float
    lon_right: float
    lat_bottom: float
    lat_top: float

def build_app(ne_shp: str, country_area_path: str, static_dir: str) -> FastAPI:
    app = FastAPI()
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    country_idx = CountryIndex(ne_shp)

    def get_rows() -> List[Dict[str, Any]]:
        return load_country_area_any(country_area_path)

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

        rows = get_rows()
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
        area_list = []
        for row in rows:
            area_id = str(row.get("id", ""))
            m = AREA_ID_RE.match(area_id)
            prefix = m.group(1).upper() if m else ""

            rect = parse_wkt_rect(row.get("geometry_wkt", ""))
            if rect is None:
                continue

            matched = prefix in code_candidates
            if not matched:
                continue

            poly = {
                "type": "Polygon",
                "coordinates": [[
                    [rect["lon_left"], rect["lat_top"]],
                    [rect["lon_right"], rect["lat_top"]],
                    [rect["lon_right"], rect["lat_bottom"]],
                    [rect["lon_left"], rect["lat_bottom"]],
                    [rect["lon_left"], rect["lat_top"]],
                ]]
            }
            feature = {
                "type": "Feature",
                "properties": {
                    "id": area_id,
                    "name": row.get("name", ""),
                    "name_en": row.get("name_en", ""),
                    "reason": row.get("reason", ""),
                    "prefix": prefix,
                    "rect": rect,
                },
                "geometry": poly,
            }
            area_features.append(feature)
            area_list.append({
                "id": area_id,
                "name": row.get("name", ""),
                "name_en": row.get("name_en", ""),
                "reason": row.get("reason", ""),
                "rect": rect,
            })

        area_list.sort(key=lambda x: x["id"])

        return {
            "country": country_feature,
            "areas": {"type": "FeatureCollection", "features": area_features},
            "area_list": area_list,
            "bbox": country_bbox,
            "summary": {
                "code": country_feature["properties"]["code"],
                "name": country_feature["properties"]["name"],
                "area_count": len(area_features),
            },
        }

    @app.get("/area")
    def get_area(id: str):
        for row in get_rows():
            if str(row.get("id", "")) == id:
                rect = parse_wkt_rect(row.get("geometry_wkt", ""))
                return {
                    "id": row.get("id"),
                    "name": row.get("name", ""),
                    "name_en": row.get("name_en", ""),
                    "reason": row.get("reason", ""),
                    "geometry_wkt": row.get("geometry_wkt", ""),
                    "rect": rect,
                }
        raise HTTPException(status_code=404, detail=f"area not found: {id}")

    @app.post("/update_area_rect")
    def update_area_rect(req: AreaUpdateRequest):
        if req.lon_left >= req.lon_right:
            raise HTTPException(status_code=400, detail="lon_left must be < lon_right")
        if req.lat_bottom >= req.lat_top:
            raise HTTPException(status_code=400, detail="lat_bottom must be < lat_top")

        rows = get_rows()
        found = False
        for row in rows:
            if str(row.get("id", "")) == req.id:
                row["geometry_wkt"] = rect_to_wkt(req.lon_left, req.lon_right, req.lat_bottom, req.lat_top)
                found = True
                break

        if not found:
            raise HTTPException(status_code=404, detail=f"area not found: {req.id}")

        bak_path = save_country_area(country_area_path, rows)
        return {
            "ok": True,
            "id": req.id,
            "backup_path": bak_path,
            "geometry_wkt": rect_to_wkt(req.lon_left, req.lon_right, req.lat_bottom, req.lat_top),
        }

    return app

def build_index_html() -> str:
    lines = [
        "<!doctype html>",
        '<html lang="ja">',
        "<head>",
        '  <meta charset="utf-8" />',
        "  <title>Country Area Editor</title>",
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0" />',
        '  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="" />',
        "  <style>",
        "    body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }",
        "    #wrap { display:grid; grid-template-columns: 1fr 440px; height:100vh; }",
        "    #map { height:100%; }",
        "    #side { padding:12px; border-left:1px solid #ddd; overflow:auto; }",
        "    .row { display:flex; gap:8px; align-items:center; margin:8px 0; }",
        "    input, button, select, textarea { padding:8px; font-size:14px; }",
        "    input, select { width:100%; }",
        "    button { cursor:pointer; }",
        "    .card { background:#f6f6f6; border-radius:10px; padding:10px; margin-top:10px; }",
        "    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; white-space:pre-wrap; }",
        "    .muted { color:#666; font-size:12px; }",
        "    .small { font-size:12px; }",
        "    label { display:block; font-size:12px; color:#444; margin-bottom:4px; }",
        "    .grid2 { display:grid; grid-template-columns: 1fr 1fr; gap:8px; }",
        "    textarea { width:100%; min-height:90px; resize:vertical; }",
        "  </style>",
        "</head>",
        "<body>",
        '<div id="wrap">',
        '  <div id="map"></div>',
        '  <div id="side">',
        '    <h2 style="margin:0 0 6px;">country_area エディタ</h2>',
        '    <div class="muted">国を選び、エリアを選ぶと赤枠をドラッグ編集できます。四隅=変形、中央=移動、Save=country_area.json 更新。</div>',
        '    <div class="card">',
        '      <label>国</label>',
        '      <div class="row">',
        '        <input id="countryKey" value="JPN" placeholder="例: JPN / Japan / 日本" />',
        '        <button id="loadCountryBtn">Load</button>',
        '      </div>',
        '      <select id="countrySelect"><option value="">国一覧を読み込み中...</option></select>',
        '    </div>',
        '    <div class="card">',
        '      <label>エリア</label>',
        '      <select id="areaSelect"><option value="">先に国を読み込んでください</option></select>',
        '      <div class="small muted" id="areaMeta">未選択</div>',
        '    </div>',
        '    <div class="card">',
        '      <div class="grid2">',
        '        <div><label>lon_left</label><input id="lon_left" /></div>',
        '        <div><label>lon_right</label><input id="lon_right" /></div>',
        '        <div><label>lat_bottom</label><input id="lat_bottom" /></div>',
        '        <div><label>lat_top</label><input id="lat_top" /></div>',
        '      </div>',
        '      <div class="row" style="margin-top:10px;">',
        '        <button id="applyBtn">Apply Coordinates</button>',
        '        <button id="saveBtn">Save</button>',
        '        <button id="reloadAreaBtn">Reload Area</button>',
        '      </div>',
        '    </div>',
        '    <div class="card">',
        '      <label>Reason</label>',
        '      <textarea id="reasonBox" readonly></textarea>',
        '    </div>',
        '    <div class="card"><div id="out" class="mono">未表示</div></div>',
        "  </div>",
        "</div>",
        '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>',
        "<script>",
        "  const map = L.map('map').setView([20, 0], 2);",
        "  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap' }).addTo(map);",
        "  const out = document.getElementById('out');",
        "  const countryKey = document.getElementById('countryKey');",
        "  const loadCountryBtn = document.getElementById('loadCountryBtn');",
        "  const countrySelect = document.getElementById('countrySelect');",
        "  const areaSelect = document.getElementById('areaSelect');",
        "  const areaMeta = document.getElementById('areaMeta');",
        "  const reasonBox = document.getElementById('reasonBox');",
        "  const lonLeftInput = document.getElementById('lon_left');",
        "  const lonRightInput = document.getElementById('lon_right');",
        "  const latBottomInput = document.getElementById('lat_bottom');",
        "  const latTopInput = document.getElementById('lat_top');",
        "  const applyBtn = document.getElementById('applyBtn');",
        "  const saveBtn = document.getElementById('saveBtn');",
        "  const reloadAreaBtn = document.getElementById('reloadAreaBtn');",
        "  let countryLayer = null;",
        "  let areasLayer = null;",
        "  let activeRect = null;",
        "  let activeAreaId = null;",
        "  let cornerMarkers = [];",
        "  let centerMarker = null;",
        "  function escapeHtml(s){ return String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('\"','&quot;').replaceAll(\"'\",'&#039;'); }",
        "  function setStatus(msg){ out.textContent = String(msg); }",
        "  function fmt(n){ return Number(n).toFixed(6); }",
        "  function rectToBounds(r){ return [[r.lat_bottom, r.lon_left], [r.lat_top, r.lon_right]]; }",
        "  function currentRectFromInputs(){ return { lon_left: Number(lonLeftInput.value), lon_right: Number(lonRightInput.value), lat_bottom: Number(latBottomInput.value), lat_top: Number(latTopInput.value) }; }",
        "  function setInputsFromRect(r){ lonLeftInput.value = fmt(r.lon_left); lonRightInput.value = fmt(r.lon_right); latBottomInput.value = fmt(r.lat_bottom); latTopInput.value = fmt(r.lat_top); }",
        "  function normalizeRect(r){ return { lon_left: Math.min(r.lon_left, r.lon_right), lon_right: Math.max(r.lon_left, r.lon_right), lat_bottom: Math.min(r.lat_bottom, r.lat_top), lat_top: Math.max(r.lat_bottom, r.lat_top) }; }",
        "  function clearEditHandles(){ for (const m of cornerMarkers) map.removeLayer(m); cornerMarkers = []; if (centerMarker) { map.removeLayer(centerMarker); centerMarker = null; } if (activeRect) { map.removeLayer(activeRect); activeRect = null; } }",
        "  function updateActiveRectFromRect(r){ if (!activeRect) return; activeRect.setBounds(rectToBounds(r)); const corners = [[r.lat_top, r.lon_left],[r.lat_top, r.lon_right],[r.lat_bottom, r.lon_right],[r.lat_bottom, r.lon_left]]; cornerMarkers.forEach((m,i)=>m.setLatLng(corners[i])); const centerLat=(r.lat_top+r.lat_bottom)/2; const centerLon=(r.lon_left+r.lon_right)/2; if (centerMarker) centerMarker.setLatLng([centerLat, centerLon]); }",
        "  function startCornerDrag(cornerKey){ return function(ev){ ev.originalEvent.preventDefault(); map.dragging.disable(); function onMove(e){ const r=currentRectFromInputs(); let lon_left=r.lon_left, lon_right=r.lon_right, lat_bottom=r.lat_bottom, lat_top=r.lat_top; const lat=e.latlng.lat; const lon=e.latlng.lng; if(cornerKey==='tl'){ lon_left=lon; lat_top=lat; } if(cornerKey==='tr'){ lon_right=lon; lat_top=lat; } if(cornerKey==='br'){ lon_right=lon; lat_bottom=lat; } if(cornerKey==='bl'){ lon_left=lon; lat_bottom=lat; } const nr=normalizeRect({lon_left,lon_right,lat_bottom,lat_top}); setInputsFromRect(nr); updateActiveRectFromRect(nr);} function onUp(){ map.dragging.enable(); map.off('mousemove', onMove); map.off('mouseup', onUp);} map.on('mousemove', onMove); map.on('mouseup', onUp); }; }",
        "  function startMoveDrag(){ return function(ev){ ev.originalEvent.preventDefault(); map.dragging.disable(); const start=ev.latlng; const base=currentRectFromInputs(); function onMove(e){ const dLat=e.latlng.lat-start.lat; const dLon=e.latlng.lng-start.lng; const nr={ lon_left: base.lon_left+dLon, lon_right: base.lon_right+dLon, lat_bottom: base.lat_bottom+dLat, lat_top: base.lat_top+dLat }; setInputsFromRect(nr); updateActiveRectFromRect(nr);} function onUp(){ map.dragging.enable(); map.off('mousemove', onMove); map.off('mouseup', onUp);} map.on('mousemove', onMove); map.on('mouseup', onUp); }; }",
        "  function renderActiveRect(r){ clearEditHandles(); activeRect = L.rectangle(rectToBounds(r), { color: '#ff0000', weight: 2, fill: false }).addTo(map); const corners=[{key:'tl',lat:r.lat_top,lon:r.lon_left},{key:'tr',lat:r.lat_top,lon:r.lon_right},{key:'br',lat:r.lat_bottom,lon:r.lon_right},{key:'bl',lat:r.lat_bottom,lon:r.lon_left}]; for (const c of corners){ const m=L.circleMarker([c.lat,c.lon],{radius:6,color:'#ff0000',fillColor:'#ffffff',fillOpacity:1,weight:2}); m.addTo(map); m.on('mousedown', startCornerDrag(c.key)); cornerMarkers.push(m);} const centerLat=(r.lat_top+r.lat_bottom)/2; const centerLon=(r.lon_left+r.lon_right)/2; centerMarker=L.circleMarker([centerLat,centerLon],{radius:5,color:'#0000ff',fillColor:'#ffffff',fillOpacity:1,weight:2}); centerMarker.addTo(map); centerMarker.on('mousedown', startMoveDrag()); }",
        "  function applyAreaInfo(info){ activeAreaId=info.id; setInputsFromRect(info.rect); reasonBox.value=info.reason||''; areaMeta.textContent=`${info.id} / ${info.name||''} / ${info.name_en||''}`; renderActiveRect(info.rect); map.fitBounds(rectToBounds(info.rect), { padding:[30,30] }); }",
        "  function clearCountryLayers(){ if(countryLayer){ map.removeLayer(countryLayer); countryLayer=null; } if(areasLayer){ map.removeLayer(areasLayer); areasLayer=null; } clearEditHandles(); activeAreaId=null; }",
        "  async function loadCountries(){ const r=await fetch('/countries'); const j=await r.json(); countrySelect.innerHTML='<option value=\"\">国一覧から選択</option>'; for(const c of j.countries){ const opt=document.createElement('option'); opt.value=c.code; opt.textContent=`${c.name} (${c.code})`; countrySelect.appendChild(opt);} }",
        "  async function loadCountry(key){ const r=await fetch(`/country_view?key=${encodeURIComponent(key)}`); if(!r.ok) throw new Error(await r.text()); const j=await r.json(); clearCountryLayers(); countryLayer=L.geoJSON(j.country,{style:{color:'#0070f3',weight:3,fill:false}}).addTo(map); areasLayer=L.geoJSON(j.areas,{ style:{color:'#ff8888',weight:1,fill:false}, onEachFeature:function(feature,layer){ const p=feature.properties||{}; layer.bindPopup(`<b>${escapeHtml(p.id||'')}</b><br>${escapeHtml(p.name||'')}<br>${escapeHtml(p.name_en||'')}`); layer.on('click', ()=>selectArea(p.id)); }}).addTo(map); const group=L.featureGroup([countryLayer,areasLayer]); map.fitBounds(group.getBounds(),{padding:[20,20]}); areaSelect.innerHTML=''; for(const a of j.area_list){ const opt=document.createElement('option'); opt.value=a.id; opt.textContent=`${a.id} / ${a.name||''}`; areaSelect.appendChild(opt); } setStatus(`国: ${j.summary.name} (${j.summary.code}) / 矩形数: ${j.summary.area_count}`); if(j.area_list.length>0){ areaSelect.value=j.area_list[0].id; await selectArea(j.area_list[0].id); } }",
        "  async function selectArea(id){ if(!id) return; const r=await fetch(`/area?id=${encodeURIComponent(id)}`); if(!r.ok) throw new Error(await r.text()); const j=await r.json(); applyAreaInfo(j); areaSelect.value=id; setStatus(`Editing: ${j.id}`); }",
        "  applyBtn.addEventListener('click', ()=>{ const r=normalizeRect(currentRectFromInputs()); setInputsFromRect(r); updateActiveRectFromRect(r); });",
        "  saveBtn.addEventListener('click', async ()=>{ if(!activeAreaId){ setStatus('area not selected'); return; } const r=normalizeRect(currentRectFromInputs()); setInputsFromRect(r); updateActiveRectFromRect(r); const resp=await fetch('/update_area_rect',{ method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ id: activeAreaId, ...r }) }); if(!resp.ok){ setStatus(await resp.text()); return; } const j=await resp.json(); setStatus(`Saved: ${j.id}\\nBackup: ${j.backup_path}`); await loadCountry(countryKey.value.trim()); await selectArea(activeAreaId); });",
        "  reloadAreaBtn.addEventListener('click', async ()=>{ if(!activeAreaId) return; await selectArea(activeAreaId); });",
        "  loadCountryBtn.addEventListener('click', async ()=>{ try{ await loadCountry(countryKey.value.trim()); } catch(e){ setStatus(String(e)); } });",
        "  countrySelect.addEventListener('change', async ()=>{ if(!countrySelect.value) return; countryKey.value=countrySelect.value; try{ await loadCountry(countrySelect.value); } catch(e){ setStatus(String(e)); } });",
        "  areaSelect.addEventListener('change', async ()=>{ if(!areaSelect.value) return; try{ await selectArea(areaSelect.value); } catch(e){ setStatus(String(e)); } });",
        "  (async ()=>{ try{ await loadCountries(); await loadCountry(countryKey.value.trim()); } catch(e){ setStatus(String(e)); } })();",
        "</script>",
        "</body>",
        "</html>",
    ]
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ne-shp", required=True, help="ne_10m_admin_0_countries.shp")
    ap.add_argument("--country-area", required=True, help="country_area.json")
    ap.add_argument("--static-dir", default="static_country_area_editor")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()

    os.makedirs(args.static_dir, exist_ok=True)
    with open(os.path.join(args.static_dir, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(build_index_html())

    app = build_app(args.ne_shp, args.country_area, args.static_dir)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
