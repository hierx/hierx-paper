# Interactive Accessibility Maps

Interactive companion maps for the hierx paper, showing population and employment
accessibility computed via hierarchical O(n log n) spatial interaction modeling.

**Live viewer**: hosted on GitHub Pages (once enabled) — `index.html` loads
tiles directly from Cloudflare R2 with no server required.

Two networks are available:
- **GB Driving** — 2.58M nodes, 3.02M edges, national scale (z5-z13)
- **London Walking** — 1.77M nodes, 1.92M edges, metro scale (z8-z14)

Features:
- Switch between networks and population/employment metrics
- Colormap selector (Inferno, Inferno reversed, Viridis, Viridis reversed)
- Dynamic color range auto-scaling to current viewport
- Click any edge for attribute values

## Basemap API key

The CARTO Positron basemap now requires an API key; without one the tiles are
served with an "API KEY REQUIRED" watermark. Keys are free for non-commercial
use (5 million tile requests per month) and can be requested without a CARTO
account at https://carto.com/basemaps/apikey. The key is public in the page
source, so restrict it to the site that serves the viewer:

- host: `hierx.github.io` (the viewer lives at
  `https://hierx.github.io/hierx-paper/interactive_maps/index.html`)
- add `localhost` if you also want local development with the key

Paste the key into `CARTO_API_KEY` near the top of the script in `index.html`.
For a one-off test without editing the file, append `?carto_key=YOUR_KEY` to
the viewer URL.

## Tile hosting

PMTiles are hosted on Cloudflare R2 (zero egress fees), in two profiles:

| Profile | File | Size | Overview tiles |
|---------|------|------|----------------|
| full | `gb_drive_accessibility.pmtiles` | 288 MB | up to 9.5 MB gzipped (26 MB raw) |
| full | `london_walk_accessibility.pmtiles` | 110 MB | up to 7.3 MB gzipped (21 MB raw) |
| lite | `gb_drive_accessibility_lite.pmtiles` | ~200 MB | at most 2 MB gzipped |
| lite | `london_walk_accessibility_lite.pmtiles` | ~70 MB | at most 2 MB gzipped |

All four live under `https://pub-0ecc092bf5574617a567a617c14b5d81.r2.dev/`.
The `full` profile keeps every edge at every zoom; its overview tiles are too
large for phone browsers, which run out of memory and drop the page. The
`lite` profile caps each tile at 2 MB by dropping the shortest edges first at
coarse zooms, keeping the long-distance road skeleton and the urban colour
pattern. Street-level zooms are identical in both.

The viewer picks `lite` on small screens, touch devices, or low-memory
devices, and `full` otherwise. Users can switch in the Detail section of the
panel, or force a profile with `?tiles=full` / `?tiles=lite`. Override the
tile URLs for local development with:

```
index.html?gb_drive=./gb_drive_accessibility.pmtiles&london_walk=./london_walk_accessibility.pmtiles
```

## Regenerating tiles

```bash
# From the repository root (hierx-paper/):

# Generate PMTiles from benchmark data (requires tippecanoe)
python interactive_maps/generate_pmtiles.py all

# Or generate individually
python interactive_maps/generate_pmtiles.py gb_drive
python interactive_maps/generate_pmtiles.py london_walk

# Serve locally (supports Range requests required by PMTiles)
python interactive_maps/serve.py
# Open http://localhost:8080
```

### Uploading to R2

```bash
# Configure rclone for R2 (one-time)
rclone config create r2 s3 \
  provider=Cloudflare \
  access_key_id=YOUR_KEY \
  secret_access_key=YOUR_SECRET \
  endpoint=https://ACCOUNT_ID.r2.cloudflarestorage.com

# Upload (use rcat to stream via stdin if direct file access fails)
for f in gb_drive_accessibility gb_drive_accessibility_lite london_walk_accessibility london_walk_accessibility_lite; do
  rclone rcat --s3-no-check-bucket r2:hierx-maps/$f.pmtiles < interactive_maps/$f.pmtiles
done
```

## Files

| File | Committed | Description |
|------|-----------|-------------|
| `index.html` | Yes | MapLibre GL JS viewer with network/metric/colormap switching |
| `generate_pmtiles.py` | Yes | NPZ → GeoJSONSeq → PMTiles pipeline |
| `serve.py` | Yes | Local HTTP server with Range request + CORS support |
| `README.md` | Yes | This file |
| `*.pmtiles` | No (.gitignore) | Tile archives, full and `_lite` profiles (regenerate or load from R2) |
| `*.geojsonl` | No (.gitignore) | Intermediate files (deleted by default) |

## Technology

- **[PMTiles](https://protomaps.com/docs/pmtiles)**: Single-file tile archive with HTTP Range request access
- **[MapLibre GL JS](https://maplibre.org/)**: Open-source WebGL vector map renderer
- **[tippecanoe](https://github.com/felt/tippecanoe)**: GeoJSON → vector tiles converter
- **[Cloudflare R2](https://www.cloudflare.com/r2/)**: Zero-egress object storage for tile hosting
- **Basemap**: CARTO Positron raster tiles (API key required; see above)
- **Colormaps**: Inferno, Inferno reversed, Viridis, Viridis reversed (matplotlib-sampled)
