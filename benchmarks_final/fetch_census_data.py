#!/usr/bin/env python3
"""Download Census source files and create derived London OA file.

Downloads:
  - England & Wales Census 2021 population (TS001) at OA level from ONS
  - England & Wales Census 2021 workplace population (WP001) from Nomis
  - Scotland Census 2022 population by Output Area from NRS

Derives:
  - london_oa_combined.csv — merged TS001 + WP001 filtered to Greater London OAs

All files are idempotent: existing files are skipped.

Usage:
    python -m benchmarks_final.fetch_census_data
"""

from __future__ import annotations

import hashlib
import logging
import time
from io import StringIO
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
CENSUS_DIR = SCRIPT_DIR.parent / "data" / "london_census"
UK_DRIVE_DIR = SCRIPT_DIR.parent / "data" / "uk_drive"

# 33 London borough local authority codes (E09000001–E09000033)
LONDON_LA_CODES = {f"E09{i:06d}" for i in range(1, 34)}


def _retry_get(
    session,
    url: str,
    *,
    params: dict | None = None,
    timeout: int = 120,
    max_retries: int = 3,
    backoff: float = 5.0,
) -> "requests.Response":
    """GET with retry on 5xx / ConnectionError, exponential backoff."""
    import requests

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, headers=_HTTP_HEADERS, timeout=timeout)
            if resp.status_code < 500:
                resp.raise_for_status()
                return resp
            # 5xx — retry
            msg = f"{resp.status_code} on attempt {attempt}/{max_retries}"
            last_exc = None
        except (requests.ConnectionError, requests.Timeout) as exc:
            msg = f"{exc.__class__.__name__} on attempt {attempt}/{max_retries}"
            last_exc = exc
        if attempt == max_retries:
            if last_exc is not None:
                raise last_exc
            resp.raise_for_status()
        wait = backoff * 2 ** (attempt - 1)
        log.warning(f"    {msg} — retrying in {wait:.0f}s ...")
        time.sleep(wait)
    raise RuntimeError("unreachable")


_HTTP_HEADERS = {
    "User-Agent": "hierx-paper/0.1 (research reproducibility; +https://github.com/hierx/hierx-paper)",
}


def _download(url: str, desc: str, timeout: int = 180, binary: bool = False) -> str | bytes:
    """Download a URL and return the response text (or bytes if binary=True)."""
    import requests

    log.info(f"  Downloading {desc} ...")
    log.info(f"    URL: {url[:120]}...")
    t0 = time.perf_counter()
    resp = requests.get(url, headers=_HTTP_HEADERS, timeout=timeout)
    resp.raise_for_status()
    dt = time.perf_counter() - t0
    log.info(f"    Downloaded {len(resp.content) / 1e6:.1f} MB in {dt:.1f}s")
    return resp.content if binary else resp.text


def _sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# Step 1: England & Wales population (TS001) — OA level
# =============================================================================


def fetch_ew_population(out_path: Path) -> pd.DataFrame:
    """Download Census 2021 TS001 (usual residents) at OA level.

    Primary source: ONS dataset download API.
    Fallback: Nomis bulk CSV API.

    Expected output columns: geography code, Residence type: Total; measures: Value
    """
    if out_path.exists():
        log.info(f"  TS001 already exists: {out_path}")
        return pd.read_csv(out_path)

    # Nomis bulk API is the primary source for OA-level data.
    # The ONS dataset download API returns LA-level, not OA-level.
    urls = [
        (
            "https://www.nomisweb.co.uk/api/v01/dataset/NM_2021_1.bulk.csv"
            "?time=latest&measures=20100&geography=TYPE150"
        ),
    ]

    df = None
    for url in urls:
        try:
            text = _download(url, "E&W TS001 population")
            df = pd.read_csv(StringIO(text))
            log.info(f"    Got {len(df)} rows, columns: {list(df.columns[:8])}")
            break
        except Exception as e:
            log.warning(f"    Failed: {e}")
            continue

    if df is None:
        raise RuntimeError(
            "Could not download TS001. Please download manually from\n"
            "https://www.nomisweb.co.uk/ and place at:\n"
            f"  {out_path}\n"
            "Expected columns: geography code, "
            "Residence type: Total; measures: Value"
        )

    # Normalise columns — Nomis bulk CSV column names vary across editions
    # TS001 columns include "Residence type: Total: All usual residents; ..."
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if "geography" in cl and "code" in cl:
            col_map[c] = "geography code"
        elif "residence type" in cl and "total" in cl and "household" not in cl:
            col_map[c] = "Residence type: Total; measures: Value"
        elif "household" in cl and "measures" in cl and "communal" not in cl:
            col_map[c] = "Residence type: Household; measures: Value"
        elif "communal" in cl and "measures" in cl:
            col_map[c] = "Residence type: Communal; measures: Value"

    if col_map:
        df = df.rename(columns=col_map)

    # Validate expected columns exist
    needed = ["geography code", "Residence type: Total; measures: Value"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        log.warning(
            f"    Missing columns {missing}. "
            f"Columns present: {list(df.columns)}"
        )
        raise RuntimeError(
            f"Downloaded TS001 CSV is missing expected columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            "Please download manually from Nomis."
        )

    # Filter to OA-level codes (E00/W00 prefix, 9 chars)
    mask = df["geography code"].astype(str).str.match(r"^[EW]\d{8}$")
    df = df[mask].copy()
    log.info(f"    Filtered to {len(df)} OA rows")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"  Saved TS001 to {out_path}")
    return df


# =============================================================================
# Step 2: England & Wales workplace population (WP001) — OA level
# =============================================================================


def fetch_ew_workplace(out_path: Path) -> pd.DataFrame:
    """Download Census 2021 workplace population (WP001) at OA level from Nomis.

    WP001 counts people by their *workplace* OA (not residence).  The dataset
    is only available as a bulk zip download, not via the Nomis query API.

    Expected output columns: Output Areas Code, Count
    """
    if out_path.exists():
        log.info(f"  WP001 already exists: {out_path}")
        return pd.read_csv(out_path)

    import io
    import zipfile

    # WP001 is only available as a bulk zip — not via the Nomis query API
    zip_url = "https://www.nomisweb.co.uk/output/census/2021/wp001.zip"

    log.info(f"  Downloading WP001 zip from {zip_url} ...")
    try:
        zip_bytes = _download(zip_url, "WP001 workplace population zip", binary=True)
    except Exception as e:
        raise RuntimeError(
            f"Could not download WP001: {e}\n"
            "Please download manually from:\n"
            f"  {zip_url}\n"
            f"Extract WP001_oa.csv and place at:\n"
            f"  {out_path}\n"
            "Expected columns: Output Areas Code, Count"
        ) from e

    # Extract WP001_oa.csv from the zip
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        oa_name = None
        for name in names:
            if name.lower().endswith("_oa.csv") or name.lower() == "wp001_oa.csv":
                oa_name = name
                break
        if oa_name is None:
            raise RuntimeError(
                f"WP001_oa.csv not found in zip.  Contents: {names}"
            )
        log.info(f"    Extracting {oa_name} from zip")
        with zf.open(oa_name) as f:
            df = pd.read_csv(f)

    log.info(f"    Got {len(df)} rows, columns: {list(df.columns)}")

    # Validate expected columns
    if "Output Areas Code" not in df.columns or "Count" not in df.columns:
        raise RuntimeError(
            f"Unexpected columns in WP001: {list(df.columns)}. "
            "Expected: Output Areas Code, Count"
        )

    # Filter to OA-level codes
    mask = df["Output Areas Code"].astype(str).str.match(r"^[EW]\d{8}$")
    df = df[mask].copy()
    log.info(f"    {len(df)} OA rows, total workplace pop: {df['Count'].sum():,}")
    log.info(f"    Max per OA: {df['Count'].max():,}, mean: {df['Count'].mean():.1f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"  Saved WP001 to {out_path}")
    return df


# =============================================================================
# Step 3: Scotland Census 2022 — OA level
# =============================================================================


def fetch_scotland_census(out_path: Path) -> pd.DataFrame:
    """Download Scotland Census 2022 OA-level population from NRS.

    Downloads the bulk OA topic tables zip from scotlandscensus.gov.uk
    and extracts UV101b (usual resident population by sex by age).

    Saves to data/uk_drive/scotland_census.csv with columns:
        oa_code, population, workplace_population
    """
    if out_path.exists():
        log.info(f"  Scotland census already exists: {out_path}")
        return pd.read_csv(out_path)

    import io
    import zipfile

    log.info("  Downloading Scotland Census 2022 OA data ...")

    # Bulk OA topic tables from scotlandscensus.gov.uk (~74 MB zip)
    zip_url = (
        "https://www.scotlandscensus.gov.uk/media/"
        "zz85kfinmf97whklasd98gfkadft5hj4f_Topic2H_20241120_1747/"
        "Census-2022-Output-Area-v1.zip"
    )

    try:
        zip_bytes = _download(zip_url, "Scotland Census 2022 OA zip", binary=True)
    except Exception as e:
        log.warning(
            f"  Scotland Census download failed: {e}\n"
            "  This is only needed for GB-wide drive accessibility.\n"
            "  Download manually from:\n"
            f"    {zip_url}\n"
            "  Extract UV101b CSV and place derived data at:\n"
            f"    {out_path}\n"
            "  Expected columns: oa_code, population, workplace_population"
        )
        return pd.DataFrame(columns=["oa_code", "population", "workplace_population"])

    # Extract UV101b (usual resident population) from the zip
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        uv101_name = None
        for name in zf.namelist():
            if "UV101" in name and name.endswith(".csv"):
                uv101_name = name
                break
        if uv101_name is None:
            raise RuntimeError(
                f"UV101 population CSV not found in zip. Contents: {zf.namelist()[:10]}"
            )

        log.info(f"    Extracting {uv101_name}")
        with zf.open(uv101_name) as f:
            # Multi-header format: 6 header lines, then data rows
            # Col 0 = OA code, Col 1 = total population ("All people", "Total")
            df = pd.read_csv(
                f, skiprows=6, header=None, na_values=["-"], low_memory=False,
            )

    # Parse: first col = OA code, second col = total population
    df = df.iloc[:, [0, 1]]
    df.columns = ["oa_code", "population"]
    df = df.dropna(subset=["oa_code"])
    df["oa_code"] = df["oa_code"].astype(str).str.strip()

    # Filter to OA-level codes (S followed by 8 digits)
    oa_mask = df["oa_code"].str.match(r"^S\d{8}$")
    df = df[oa_mask].copy()
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0).astype(int)

    # Scotland workplace population is not available at OA level;
    # estimate as fraction of residential population (Scotland average ~45%)
    df["workplace_population"] = (df["population"] * 0.45).astype(int)

    log.info(f"    {len(df):,} OAs, total pop: {df['population'].sum():,}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"  Saved Scotland census to {out_path}")
    return df


# =============================================================================
# Step 4: Derive london_oa_combined.csv
# =============================================================================


def _has_oa21(name: str) -> bool:
    """Check if name contains OA21 but NOT only as part of LSOA21/MSOA21."""
    import re
    # Remove LSOA21 and MSOA21 occurrences, then check for OA21
    cleaned = re.sub(r"[LM]SOA21", "", name)
    return "OA21" in cleaned or "Output_Area" in name


def _fetch_oa_to_lsoa(
    arcgis_base: str,
    svc_names: list[str],
    requests_mod: object,
) -> dict[str, str]:
    """Fetch OA21→LSOA21 mapping from ONS ArcGIS.

    Returns dict mapping OA21CD → LSOA21CD.
    """
    # Find a service with genuine OA21 (not just LSOA21/MSOA21)
    candidates = [n for n in svc_names if _has_oa21(n) and "LSOA" in n]
    if not candidates:
        candidates = [n for n in svc_names if _has_oa21(n)]
    if not candidates:
        raise RuntimeError(
            "Could not find OA21 service for OA→LSOA mapping.\n"
            f"  Searched {len(svc_names)} services."
        )

    svc = candidates[0]
    log.info(f"    OA→LSOA service: {svc}")
    query_url = f"{arcgis_base}/{svc}/FeatureServer/0/query"

    session = requests_mod.Session()

    # Discover fields
    resp = _retry_get(
        session, query_url,
        params={"where": "1=1", "outFields": "*", "f": "json",
                "resultRecordCount": 1},
        timeout=30,
    )
    sample = resp.json()
    fields = list(
        sample["features"][0]["attributes"].keys()
    ) if sample.get("features") else []

    oa_f = next((f for f in fields if f.upper() == "OA21CD"), None)
    lsoa_f = next(
        (f for f in fields
         if f.upper().startswith("LSOA21") and f.upper().endswith("CD")),
        None,
    )
    if not oa_f or not lsoa_f:
        raise RuntimeError(
            f"Could not find OA21CD/LSOA21CD in {fields}"
        )

    log.info(f"    Using: {oa_f}, {lsoa_f}")

    mapping: dict[str, str] = {}
    batch_size = 1000
    offset = 0

    while True:
        params = {
            "where": "1=1",
            "outFields": f"{oa_f},{lsoa_f}",
            "f": "json",
            "resultRecordCount": batch_size,
            "resultOffset": offset,
        }
        resp = _retry_get(session, query_url, params=params, timeout=120)
        data = resp.json()

        features = data.get("features", [])
        if not features:
            break

        for feat in features:
            attrs = feat.get("attributes", {})
            mapping[attrs.get(oa_f, "")] = attrs.get(lsoa_f, "")

        if len(mapping) % 20000 < batch_size:
            log.info(f"    OA→LSOA: {len(mapping):,} mappings ...")

        if len(features) < batch_size:
            break
        offset += batch_size

    log.info(f"    OA→LSOA complete: {len(mapping):,} OAs")
    return mapping


def fetch_oa_to_la_lookup(cache_path: Path | None = None) -> pd.DataFrame:
    """Download OA-to-LA lookup from ONS Open Geography Portal.

    The service name changes when ONS publishes new editions, so we
    discover the correct one dynamically from the ArcGIS services list.

    Returns DataFrame with columns: oa21cd, lad22cd
    """
    import requests

    if cache_path is None:
        cache_path = CENSUS_DIR / "oa21_to_lad_lookup.csv"
    if cache_path.exists():
        log.info(f"  OA-to-LA lookup cached: {cache_path}")
        return pd.read_csv(cache_path)

    log.info("  Fetching OA-to-LA lookup from ONS ArcGIS (~190k rows, may take a few minutes) ...")

    session = requests.Session()

    arcgis_base = (
        "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services"
    )

    # Step 1: list all services and find OA21-to-LAD lookup
    resp = _retry_get(session, arcgis_base, params={"f": "json"}, timeout=60)
    services = resp.json().get("services", [])
    log.info(f"    ArcGIS portal has {len(services)} services")

    # Find a service whose name contains OA21 and LAD (the lookup table).
    # Prefer services with both OA21 and LAD in the name; fall back to
    # OA21+LU (lookup), then LSOA21+LAD (we can chain OA→LSOA→LAD).
    svc_names = [s["name"] for s in services]

    candidates = [n for n in svc_names if _has_oa21(n) and "LAD" in n]
    if not candidates:
        candidates = [n for n in svc_names if _has_oa21(n) and "LU" in n]
    if not candidates:
        # Fall back to LSOA21→LAD (every OA maps to one LSOA)
        candidates = [n for n in svc_names if "LSOA21" in n and "LAD" in n]
    if not candidates:
        raise RuntimeError(
            "Could not find OA/LSOA-to-LAD lookup service on ONS ArcGIS.\n"
            f"  Searched {len(services)} services.\n"
            "  Please download the OA-to-LAD lookup manually from\n"
            "  https://geoportal.statistics.gov.uk/ and save to:\n"
            f"  {cache_path}\n"
            "  Expected columns: oa21cd, lad22cd"
        )

    service_name = candidates[0]
    log.info(f"    Found lookup service: {service_name}")

    # Step 2: page through the FeatureServer to get all OA→LAD mappings
    query_url = f"{arcgis_base}/{service_name}/FeatureServer/0/query"

    # First, discover available field names
    fields_resp = _retry_get(
        session, query_url,
        params={"where": "1=1", "outFields": "*", "f": "json",
                "resultRecordCount": 1},
        timeout=30,
    )
    sample = fields_resp.json()
    if "features" in sample and sample["features"]:
        available_fields = list(sample["features"][0].get("attributes", {}).keys())
        log.info(f"    Available fields: {available_fields[:10]}")
    else:
        available_fields = []

    # Find the OA/LSOA and LAD field names (case may vary)
    oa_field = next(
        (f for f in available_fields if f.upper().startswith("OA21")), None
    )
    lsoa_field = next(
        (f for f in available_fields
         if f.upper().startswith("LSOA21") and f.upper().endswith("CD")),
        None,
    )
    lad_field = next(
        (f for f in available_fields
         if f.upper().startswith("LAD") and "CD" in f.upper()),
        None,
    )

    if not lad_field:
        raise RuntimeError(
            f"Could not identify LAD field in {available_fields}"
        )

    # If we have OA21 directly, use it; otherwise use LSOA21→LAD and
    # derive OA→LAD from OA code prefix (OA codes nest inside LSOAs)
    use_lsoa_path = oa_field is None
    geo_field = oa_field or lsoa_field
    if not geo_field:
        raise RuntimeError(
            f"Could not identify OA or LSOA field in {available_fields}"
        )

    log.info(f"    Using fields: {geo_field}, {lad_field}")

    all_rows: list[dict] = []
    # ArcGIS FeatureServer default max is 1000 records per page
    batch_size = 1000
    offset = 0

    while True:
        params = {
            "where": "1=1",
            "outFields": f"{geo_field},{lad_field}",
            "f": "json",
            "resultRecordCount": batch_size,
            "resultOffset": offset,
        }
        resp = _retry_get(session, query_url, params=params, timeout=120)
        data = resp.json()

        features = data.get("features", [])
        if not features:
            break

        for feat in features:
            attrs = feat.get("attributes", {})
            all_rows.append({
                "oa21cd": attrs.get(geo_field, ""),
                "lad22cd": attrs.get(lad_field, ""),
            })

        if len(all_rows) % 10000 < batch_size:
            log.info(f"    Fetched {len(all_rows):,} / ~190,000 OA-to-LA mappings")

        if len(features) < batch_size:
            break
        offset += batch_size

    if not all_rows:
        raise RuntimeError("OA-to-LA lookup returned 0 rows")

    raw_lookup = pd.DataFrame(all_rows)

    if use_lsoa_path:
        # We have LSOA→LAD; need to map OA→LSOA→LAD.
        log.info("    Service has LSOA-level only; finding OA→LSOA mapping ...")
        lsoa_to_lad = dict(
            zip(raw_lookup["oa21cd"], raw_lookup["lad22cd"], strict=False)
        )
        oa_to_lsoa = _fetch_oa_to_lsoa(arcgis_base, svc_names, requests)
        lookup = pd.DataFrame([
            {"oa21cd": oa, "lad22cd": lsoa_to_lad.get(lsoa, "")}
            for oa, lsoa in oa_to_lsoa.items()
            if lsoa_to_lad.get(lsoa)
        ])
    else:
        lookup = raw_lookup

    lookup = pd.DataFrame(all_rows)
    log.info(f"  OA-to-LA lookup complete: {len(lookup):,} rows")

    # Cache for reproducibility
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(cache_path, index=False)
    log.info(f"  Cached to {cache_path}")

    return lookup


def derive_london_oa_combined(
    pop_path: Path,
    wp_path: Path,
    out_path: Path,
) -> pd.DataFrame:
    """Merge TS001 + WP001 and filter to Greater London OAs.

    Output columns: oa21cd, population, pop_household, pop_communal, workplace_population
    """
    if out_path.exists():
        log.info(f"  london_oa_combined already exists: {out_path}")
        return pd.read_csv(out_path)

    log.info("  Deriving london_oa_combined.csv ...")

    # Load population — columns were normalised by fetch_ew_population
    pop_df = pd.read_csv(pop_path)

    # Build rename map for whatever columns are present
    rename = {"geography code": "oa21cd"}
    for c in pop_df.columns:
        cl = c.lower()
        if "total" in cl and "residence" in cl:
            rename[c] = "population"
        elif "household" in cl and "measures" in cl and "communal" not in cl:
            rename[c] = "pop_household"
        elif "communal" in cl and "measures" in cl:
            rename[c] = "pop_communal"
    pop_df = pop_df.rename(columns=rename)

    pop_cols = ["oa21cd", "population"]
    if "pop_household" in pop_df.columns:
        pop_cols.append("pop_household")
    if "pop_communal" in pop_df.columns:
        pop_cols.append("pop_communal")

    pop_df = pop_df[pop_cols].copy()

    # Load workplace
    wp_df = pd.read_csv(wp_path)
    wp_df = wp_df.rename(columns={
        "Output Areas Code": "oa21cd",
        "Count": "workplace_population",
    })[["oa21cd", "workplace_population"]]

    # Merge
    census = pop_df.merge(wp_df, on="oa21cd", how="outer")
    census["population"] = census["population"].fillna(0).astype(int)
    census["workplace_population"] = census["workplace_population"].fillna(0).astype(int)

    # Fill household/communal if missing
    if "pop_household" not in census.columns:
        census["pop_household"] = census["population"]
    else:
        census["pop_household"] = census["pop_household"].fillna(0).astype(int)
    if "pop_communal" not in census.columns:
        census["pop_communal"] = 0
    else:
        census["pop_communal"] = census["pop_communal"].fillna(0).astype(int)

    log.info(f"  Merged census: {len(census):,} OAs total")

    # Filter to London: get OA-to-LA lookup and keep London LAs
    lookup = fetch_oa_to_la_lookup()
    london_oas = set(
        lookup[lookup["lad22cd"].isin(LONDON_LA_CODES)]["oa21cd"].tolist()
    )
    log.info(f"  London OAs from lookup: {len(london_oas):,}")

    london_census = census[census["oa21cd"].isin(london_oas)].copy()
    log.info(
        f"  London filtered: {len(london_census):,} OAs, "
        f"pop={london_census['population'].sum():,}, "
        f"wp={london_census['workplace_population'].sum():,}"
    )

    # Reorder columns
    london_census = london_census[
        ["oa21cd", "population", "pop_household", "pop_communal", "workplace_population"]
    ].sort_values("oa21cd").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    london_census.to_csv(out_path, index=False)
    log.info(f"  Saved to {out_path}")
    return london_census


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    log.info("=" * 60)
    log.info("Census data fetch pipeline")
    log.info("=" * 60)

    CENSUS_DIR.mkdir(parents=True, exist_ok=True)
    UK_DRIVE_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # Step 1: E&W Population (TS001)
    log.info("")
    log.info("--- Step 1: E&W Population (TS001) ---")
    pop_path = CENSUS_DIR / "census2021-ts001-oa.csv"
    fetch_ew_population(pop_path)

    # Step 2: E&W Workplace (WP001)
    log.info("")
    log.info("--- Step 2: E&W Workplace (WP001) ---")
    wp_path = CENSUS_DIR / "WP001_oa.csv"
    fetch_ew_workplace(wp_path)

    # Step 3: Scotland Census 2022
    log.info("")
    log.info("--- Step 3: Scotland Census 2022 ---")
    scot_path = UK_DRIVE_DIR / "scotland_census.csv"
    fetch_scotland_census(scot_path)

    # Step 4: Derive london_oa_combined.csv
    log.info("")
    log.info("--- Step 4: Derive london_oa_combined.csv ---")
    london_path = CENSUS_DIR / "london_oa_combined.csv"
    derive_london_oa_combined(pop_path, wp_path, london_path)

    dt = time.perf_counter() - t0
    log.info("")
    log.info(f"Census fetch complete in {dt:.1f}s")
    log.info(f"  {pop_path}")
    log.info(f"  {wp_path}")
    log.info(f"  {scot_path}")
    log.info(f"  {london_path}")


if __name__ == "__main__":
    main()
