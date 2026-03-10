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

## Tile hosting

PMTiles are hosted on Cloudflare R2 (zero egress fees):

- `https://pub-0ecc092bf5574617a567a617c14b5d81.r2.dev/gb_drive_accessibility.pmtiles` (288 MB)
- `https://pub-0ecc092bf5574617a567a617c14b5d81.r2.dev/london_walk_accessibility.pmtiles` (110 MB)

The viewer (`index.html`) loads these URLs by default. Override with query params
for local development:

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
rclone rcat --s3-no-check-bucket r2:hierx-maps/gb_drive_accessibility.pmtiles < interactive_maps/gb_drive_accessibility.pmtiles
rclone rcat --s3-no-check-bucket r2:hierx-maps/london_walk_accessibility.pmtiles < interactive_maps/london_walk_accessibility.pmtiles
```

## Files

| File | Committed | Description |
|------|-----------|-------------|
| `index.html` | Yes | MapLibre GL JS viewer with network/metric/colormap switching |
| `generate_pmtiles.py` | Yes | NPZ → GeoJSONSeq → PMTiles pipeline |
| `serve.py` | Yes | Local HTTP server with Range request + CORS support |
| `README.md` | Yes | This file |
| `*.pmtiles` | No (.gitignore) | Tile archives (regenerate or load from R2) |
| `*.geojsonl` | No (.gitignore) | Intermediate files (deleted by default) |

## Technology

- **[PMTiles](https://protomaps.com/docs/pmtiles)**: Single-file tile archive with HTTP Range request access
- **[MapLibre GL JS](https://maplibre.org/)**: Open-source WebGL vector map renderer
- **[tippecanoe](https://github.com/felt/tippecanoe)**: GeoJSON → vector tiles converter
- **[Cloudflare R2](https://www.cloudflare.com/r2/)**: Zero-egress object storage for tile hosting
- **Basemap**: CARTO Positron
- **Colormaps**: Inferno, Inferno reversed, Viridis, Viridis reversed (matplotlib-sampled)
