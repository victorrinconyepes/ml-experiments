import os
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Polygon
from shapely.geometry.base import BaseGeometry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

BEARER_TOKEN = os.getenv("BEARER_TOKEN_VEXCEL")

# ================================
# CONFIG ENTRADA / COLUMNAS
# ================================
GPKG_SPLITS_PATH = "/ceph04/ml/property_damage_elements/regression/geopackages_image_candidates/ensure_deduplication/selected_imagery.gpkg"
SPLIT_LAYERS = ["train", "val"]
AOI_COLUMN = "child_aoi"
ID_COLUMN = "id"

# Lista de columnas que quieres copiar a la salida.
# Si se deja vacía => se copian todas (excepto geometry).
# Pon aquí las que quieras controlar explícitamente, por ejemplo:
COLS = ["id", "child_aoi", "area_m2"]
ALL_COLS = [
    "roof_condition_rust_percent",
    "roof_discoloration_algae_staining_percen",
    "roof_discoloration_vent_staining_percent",
    "roof_discoloration_water_pooling_percent",
    "roof_discoloration_debris_percent",
    "missing_material_percent",
    "patch_percent",
    "roof_condition_structural_damage_percent"
]
ATTR_COLUMNS = COLS + ALL_COLS
# ATTR_COLUMNS: List[str] = []

# ================================
# CONFIG SALIDA / DESCARGA
# ================================
OUTPUT_IMAGE_DIR = "/ceph04/ml/property_damage_elements/regression/geopackages_image_candidates/ensure_deduplication/"
LOG_CSV = os.path.join(OUTPUT_IMAGE_DIR, "imagery_download_log.csv")

IMAGE_FORMAT = "png"
BBOX_BUFFER_PCT = 0.10
FORCE_SQUARE = True
MAX_SIDE_DEG = None
MIN_SIDE_DEG = None

VEXCEL_BASE_URL = "https://api.vexcelgroup.com/v2/ortho/extract"

MAX_RETRIES = 3
RETRY_SLEEP_SEC = 3

ENABLE_CONCURRENCY = True
MAX_WORKERS = 12

PER_REQUEST_SLEEP = 0.15
SKIP_IF_EXISTS = True      # Si True: no se vuelve a descargar, pero SÍ se agrega la fila al GPKG
DRY_RUN = False

LOG_LEVEL = logging.INFO

GPKG_OUTPUT_PATH = os.path.join(GPKG_SPLITS_PATH, "data.gpkg")
OUTPUT_LAYER_NAME = "imagery_downloads"

DOWNLOAD_RECORDS: List[Dict[str, Any]] = []

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("imagery_downloader")


# =========================================
# UTILIDADES
# =========================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def compute_bbox_polygon(geom: BaseGeometry,
                         buffer_pct: float = 0.0,
                         force_square: bool = False,
                         min_side: Optional[float] = None,
                         max_side: Optional[float] = None) -> Polygon:
    if geom is None or geom.is_empty:
        raise ValueError("Geometría vacía o inválida.")
    minx, miny, maxx, maxy = geom.bounds
    width = maxx - minx
    height = maxy - miny
    if width == 0 or height == 0:
        logger.warning("Geometría degenerada (area 0). Generando bbox mínima.")
        width = height = 1e-6
        minx -= width / 2; maxx += width / 2
        miny -= height / 2; maxy += height / 2
    if buffer_pct > 0:
        extra_w = width * buffer_pct
        extra_h = height * buffer_pct
        minx -= extra_w / 2; maxx += extra_w / 2
        miny -= extra_h / 2; maxy += extra_h / 2
        width = maxx - minx; height = maxy - miny
    if force_square:
        side = max(width, height)
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2
        half = side / 2
        minx, maxx = cx - half, cx + half
        miny, maxy = cy - half, cy + half
    if min_side and (maxx - minx) < min_side:
        diff = (min_side - (maxx - minx)) / 2
        minx -= diff; maxx += diff; miny -= diff; maxy += diff
    if max_side and (maxx - minx) > max_side:
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2
        half = max_side / 2
        minx, maxx = cx - half, cx + half
        miny, maxy = cy - half, cy + half
    return box(minx, miny, maxx, maxy)

def bbox_polygon_to_wkt(poly: Polygon) -> str:
    return poly.wkt

def build_output_path(split: str, child_aoi: str, feature_id: str) -> Path:
    return Path(OUTPUT_IMAGE_DIR) / split / child_aoi / f"{feature_id}.{IMAGE_FORMAT}"

def write_image(content: bytes, out_path: Path):
    ensure_dir(out_path.parent)
    with open(out_path, "wb") as f:
        f.write(content)

def init_log_csv(path: str):
    if not Path(path).exists():
        with open(path, "w", encoding="utf-8") as f:
            f.write("id,split,child_aoi,status,http_status,attempts,output_path,message\n")

def append_log(path: str,
               feature_id: str,
               split: str,
               child_aoi: str,
               status: str,
               http_status: Optional[int],
               attempts: int,
               output_path: Optional[str],
               message: str):
    safe_msg = message.replace("\n", " ").replace(",", ";")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{feature_id},{split},{child_aoi},{status},{http_status or ''},{attempts},{output_path or ''},{safe_msg}\n")


def request_vexcel_image(bbox_wkt: str, child_aoi: str) -> Tuple[bool, Optional[bytes], Dict[str, Any], Optional[int], str]:
    params = {
        "layer": 'urban',
        "wkt": bbox_wkt,
        "srid": "4326",
        "collection": child_aoi,
        "product-type": "final-ortho",
        "zoom": 21,
        "crop": "clip",
        "attribution": False,
        "image-format": IMAGE_FORMAT,
    }
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

    if DRY_RUN:
        meta = {"http_status": None, "headers": {}, "attempt": 0, "params_sent": params, "collection": child_aoi}
        return True, b"", meta, None, "DRY_RUN"

    if not BEARER_TOKEN:
        return False, None, {"error": "Missing BEARER_TOKEN_VEXCEL env var"}, None, "No token"

    last_exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(VEXCEL_BASE_URL, params=params, headers=headers, timeout=30)
            http_status = resp.status_code
            if http_status == 200:
                meta = {
                    "http_status": http_status,
                    "headers": dict(resp.headers),
                    "attempt": attempt,
                    "params_sent": params,
                    "collection": child_aoi
                }
                return True, resp.content, meta, http_status, "OK"
            else:
                msg = f"HTTP {http_status}: {resp.text[:300]}"
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_SLEEP_SEC)
                else:
                    return False, None, {"error": msg}, http_status, msg
        except Exception as ex:
            last_exception = ex
            trace = "".join(traceback.format_exception_only(type(ex), ex)).strip()
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SEC)
            else:
                return False, None, {"exception": trace}, None, f"Exception: {trace}"
    return False, None, {"exception": str(last_exception)}, None, "Unknown failure"


def extract_attributes(row) -> Dict[str, Any]:
    """
    Devuelve un dict con los atributos a guardar (excluye geometry).
    Si ATTR_COLUMNS está vacío => usa todas las columnas no-geometry.
    """
    if ATTR_COLUMNS:
        cols = [c for c in ATTR_COLUMNS if c in row.index and c != "geometry"]
    else:
        cols = [c for c in row.index if c != "geometry"]
    data = {}
    for c in cols:
        # .get puede devolver tipos numpy; se pueden convertir si prefieres.
        val = row.get(c, None)
        # Conversión opcional de tipos no serializables
        if isinstance(val, (pd.Timestamp,)):
            val = val.isoformat()
        data[c] = val
    return data


def register_record(feature_id, split, child_aoi, bbox_wkt, out_path, geom,
                    attempt, http_status, row):
    metadata_row = {
        "feature_id": feature_id,
        "split": split,
        "child_aoi": child_aoi,
        "bbox_wkt": bbox_wkt,
        "output_path": str(out_path),
        "attempt": attempt,
        "http_status": http_status,
        "geometry": geom
    }
    # Añadir atributos dinámicos
    metadata_row.update(extract_attributes(row))
    DOWNLOAD_RECORDS.append(metadata_row)


def process_feature(row, split: str):
    feature_id = str(row[ID_COLUMN])
    child_aoi = str(row[AOI_COLUMN])
    geom = row.geometry
    out_path = build_output_path(split, child_aoi, feature_id)

    try:
        bbox_poly = compute_bbox_polygon(
            geom,
            buffer_pct=BBOX_BUFFER_PCT,
            force_square=FORCE_SQUARE,
            min_side=MIN_SIDE_DEG,
            max_side=MAX_SIDE_DEG
        )
        bbox_wkt = bbox_polygon_to_wkt(bbox_poly)
    except Exception as e:
        append_log(LOG_CSV, feature_id, split, child_aoi, "bbox_error", None, 0, None, f"BBox error: {e}")
        return

    if SKIP_IF_EXISTS and out_path.exists():
        register_record(
            feature_id, split, child_aoi, bbox_wkt, out_path, geom,
            attempt=0, http_status=None, row=row
        )
        append_log(LOG_CSV, feature_id, split, child_aoi, "skipped_exists", None, 0, str(out_path), "File exists")
        return

    success, img_bytes, meta, http_status, message = request_vexcel_image(bbox_wkt, child_aoi)

    if success:
        if not DRY_RUN:
            write_image(img_bytes, out_path)
        register_record(
            feature_id, split, child_aoi, bbox_wkt, out_path, geom,
            attempt=meta.get("attempt"),
            http_status=http_status,
            row=row
        )
        append_log(LOG_CSV, feature_id, split, child_aoi, "success", http_status, meta.get("attempt", 1), str(out_path), message)
    else:
        append_log(LOG_CSV, feature_id, split, child_aoi, "failed", http_status, meta.get("attempt", MAX_RETRIES), None, message)

    if PER_REQUEST_SLEEP > 0:
        time.sleep(PER_REQUEST_SLEEP)


def load_split_layer(gpkg_path: str, layer_name: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    if gdf.crs is None or str(gdf.crs).lower() != "epsg:4326":
        logger.warning(f"Capa {layer_name} no en EPSG:4326. Forzando.")
        gdf.set_crs("EPSG:4326", inplace=True)
    return gdf

def process_layer(gpkg_path: str, layer_name: str):
    logger.info(f"Procesando capa: {layer_name}")
    gdf = load_split_layer(gpkg_path, layer_name)
    for col in [AOI_COLUMN, ID_COLUMN]:
        if col not in gdf.columns:
            raise ValueError(f"La capa {layer_name} no contiene la columna requerida '{col}'")
    total = len(gdf)
    logger.info(f"{layer_name}: {total} features")
    pbar = tqdm(total=total, desc=f"{layer_name}", unit="img")
    if ENABLE_CONCURRENCY:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_feature, row, layer_name) for _, row in gdf.iterrows()]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    logger.error(f"Error future en capa {layer_name}: {e}")
                pbar.update(1)
    else:
        for _, row in gdf.iterrows():
            process_feature(row, layer_name)
            pbar.update(1)
    pbar.close()
    logger.info(f"Fin capa: {layer_name}")


def flush_download_records():
    global DOWNLOAD_RECORDS
    if not DOWNLOAD_RECORDS:
        logger.warning("No hay registros para escribir.")
        return
    df = pd.DataFrame(DOWNLOAD_RECORDS)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    if Path(GPKG_OUTPUT_PATH).exists():
        Path(GPKG_OUTPUT_PATH).unlink()
    gdf.to_file(GPKG_OUTPUT_PATH, layer=OUTPUT_LAYER_NAME, driver="GPKG")
    logger.info(f"Escrito {len(gdf)} filas en capa '{OUTPUT_LAYER_NAME}' ({GPKG_OUTPUT_PATH})")
    DOWNLOAD_RECORDS = []


def main():
    logger.info("==== Inicio descarga de imágenes (acumulando en memoria) ====")
    ensure_dir(Path(OUTPUT_IMAGE_DIR))
    init_log_csv(LOG_CSV)

    if not BEARER_TOKEN and not DRY_RUN:
        logger.warning("No se encontró BEARER_TOKEN_VEXCEL (las peticiones fallarán).")

    for layer in SPLIT_LAYERS:
        process_layer(GPKG_SPLITS_PATH, layer)

    flush_download_records()

    logger.info("==== Proceso completado ====")
    if DRY_RUN:
        logger.info("Modo DRY_RUN activo (no se descargaron imágenes).")

if __name__ == "__main__":
    main()