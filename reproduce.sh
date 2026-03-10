#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — orchestrate hierx-paper reproducibility pipeline
#
# Usage:
#   ./reproduce.sh test           Run hierx test suite
#   ./reproduce.sh data           Fetch archived data from Zenodo
#   ./reproduce.sh figures        Regenerate paper figures from pre-computed JSON
#   ./reproduce.sh benchmarks     Rerun all synthetic benchmarks (~30-60 min)
#   ./reproduce.sh london         Run London network benchmarks
#   ./reproduce.sh accessibility  Compute accessibility maps (GB drive + London walk)
#   ./reproduce.sh reproduce      Full reproduction from Zenodo data + verify results
#   ./reproduce.sh data-fetch     Download all source data (census + networks) from origin
#   ./reproduce.sh maps           Generate interactive PMTiles maps
#   ./reproduce.sh all            Full pipeline from source (data-fetch + benchmarks)
#
# The default mode (no arguments) is "figures".
# =============================================================================
set -euo pipefail

# macOS compatibility: sha256sum may not exist
if ! command -v sha256sum &>/dev/null; then
    sha256sum() { shasum -a 256 "$@"; }
fi

MODE="${1:-figures}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Warn if /dev/shm is too small for parallel hierarchy builds
if [[ -d /dev/shm ]]; then
    shm_mb=$(df -m /dev/shm 2>/dev/null | awk 'NR==2 {print $4}')
    if [[ -n "$shm_mb" && "$shm_mb" -lt 256 ]]; then
        echo "WARNING: /dev/shm has only ${shm_mb} MB free (256 MB recommended)."
        echo "  Parallel hierarchy builds may fall back to slower fork-based pooling."
        echo "  Fix: docker run --shm-size=256m ..."
        echo ""
    fi
fi
ZENODO_DOI="${ZENODO_DOI:-10.5281/zenodo.19062193}"
ZENODO_SANDBOX="${ZENODO_SANDBOX:-false}"
ZENODO_TOKEN="${ZENODO_TOKEN:-}"
MANIFEST="${SCRIPT_DIR}/zenodo_manifest.txt"

log() {
    echo ""
    echo "========================================"
    echo "  $1"
    echo "========================================"
}

# ---------------------------------------------------------------------------
# fetch_zenodo — download archived data files from Zenodo
# ---------------------------------------------------------------------------
fetch_zenodo() {
    log "Fetching data from Zenodo (DOI: ${ZENODO_DOI})"


    if [[ ! -f "$MANIFEST" ]]; then
        echo "ERROR: Manifest file not found: ${MANIFEST}"
        return 1
    fi

    # Extract numeric record ID from DOI (e.g. 10.5281/zenodo.1234567 → 1234567)
    local record_id
    record_id="${ZENODO_DOI##*/zenodo.}"

    # Resolve the record URL and get the download links
    local zenodo_base="https://zenodo.org"
    if [[ "$ZENODO_SANDBOX" == "true" ]]; then
        zenodo_base="https://sandbox.zenodo.org"
        echo "  (using Zenodo sandbox)"
    fi
    # Build query string with optional access token
    local qs=""
    if [[ -n "$ZENODO_TOKEN" ]]; then
        qs="?access_token=${ZENODO_TOKEN}"
        echo "  (using access token)"
    fi

    local api_url="${zenodo_base}/api/records/${record_id}${qs}"
    echo "Querying Zenodo API: ${zenodo_base}/api/records/${record_id}"

    local record_json
    record_json=$(curl -fsSL "$api_url") || {
        echo "ERROR: Failed to fetch Zenodo record ${record_id}"
        return 1
    }

    local download_count=0
    local skip_count=0
    local fail_count=0

    while IFS=$'\t' read -r zenodo_name local_path expected_sha; do
        # Skip comments and blank lines
        [[ -z "$zenodo_name" || "$zenodo_name" == \#* ]] && continue

        local full_path="${SCRIPT_DIR}/${local_path}"

        # Skip files that already exist (with optional checksum verification)
        if [[ -f "$full_path" ]]; then
            if [[ "$expected_sha" != "TODO" && -n "$expected_sha" ]]; then
                local actual_sha
                actual_sha=$(sha256sum "$full_path" | awk '{print $1}')
                if [[ "$actual_sha" != "$expected_sha" ]]; then
                    echo "  MISMATCH: ${local_path} — re-downloading"
                    echo "    expected: ${expected_sha}"
                    echo "    actual:   ${actual_sha}"
                    rm -f "$full_path"
                fi
            fi
        fi

        # If file still exists after checksum check, skip it
        if [[ -f "$full_path" ]]; then
            echo "  SKIP (exists, verified): ${local_path}"
            skip_count=$((skip_count + 1))
            continue
        fi

        # Find the download URL for this file from the record JSON
        local file_url
        file_url=$(echo "$record_json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for f in data.get('files', []):
    if f['key'] == '${zenodo_name}':
        print(f['links']['self'])
        break
" 2>/dev/null) || true

        if [[ -z "$file_url" ]]; then
            echo "  WARN: File '${zenodo_name}' not found in Zenodo record"
            fail_count=$((fail_count + 1))
            continue
        fi

        # Create parent directory
        mkdir -p "$(dirname "$full_path")"

        echo "  GET: ${zenodo_name} → ${local_path}"
        # Append access token to download URL if set
        local dl_url="$file_url"
        if [[ -n "$ZENODO_TOKEN" ]]; then
            dl_url="${file_url}?access_token=${ZENODO_TOKEN}"
        fi
        if curl -fSL -o "$full_path" "$dl_url"; then
            # Verify checksum if not a placeholder
            if [[ "$expected_sha" != "TODO" && -n "$expected_sha" ]]; then
                local actual_sha
                actual_sha=$(sha256sum "$full_path" | awk '{print $1}')
                if [[ "$actual_sha" != "$expected_sha" ]]; then
                    echo "  ERROR: Checksum mismatch for ${local_path}"
                    echo "    expected: ${expected_sha}"
                    echo "    actual:   ${actual_sha}"
                    rm -f "$full_path"
                    fail_count=$((fail_count + 1))
                    continue
                fi
                echo "  OK: checksum verified"
            else
                echo "  WARN: checksum is TODO — skipping verification"
            fi
            download_count=$((download_count + 1))
        else
            echo "  ERROR: Download failed for ${zenodo_name}"
            fail_count=$((fail_count + 1))
        fi
    done < "$MANIFEST"

    echo ""
    echo "Zenodo fetch complete: ${download_count} downloaded, ${skip_count} skipped, ${fail_count} failed."
    [[ "$fail_count" -eq 0 ]] || return 1
}

# ---------------------------------------------------------------------------
# Pyrosm environment helper
# ---------------------------------------------------------------------------
PYROSM_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.mamba}"

run_pyrosm() {
    # Run a command in the pyrosm micromamba environment if available
    if [[ -x /tmp/bin/micromamba ]]; then
        MAMBA_ROOT_PREFIX="$PYROSM_PREFIX" /tmp/bin/micromamba run -n pyrosm "$@"
    elif command -v micromamba &>/dev/null; then
        MAMBA_ROOT_PREFIX="$PYROSM_PREFIX" micromamba run -n pyrosm "$@"
    else
        "$@"
    fi
}

# ---------------------------------------------------------------------------
# data-fetch — download all source data from origin (census + OSM networks)
# ---------------------------------------------------------------------------
run_data_fetch() {
    log "Phase 1: Census data (E&W + Scotland + London OA combined)"
    python -m benchmarks_final.fetch_census_data

    log "Phase 2: London drive networks (OSMnx)"
    python -m benchmarks_final.fetch_london_networks

    log "Phase 3: London walking networks (OSMnx, incl. 1M)"
    python -m benchmarks_final.fetch_london_walking

    log "Phase 4: Build London population activity"
    python -m benchmarks_final.build_population_activity

    log "Phase 5: GB drive network (requires pyrosm conda env)"
    run_pyrosm python -m benchmarks_final.fetch_uk_drive

    log "Phase 6: Build UK population activity (requires pyrosm conda env)"
    run_pyrosm python -m benchmarks_final.build_uk_population_activity

    echo ""
    echo "All source data downloaded and population activity built."
}

# ---------------------------------------------------------------------------
# test — run the hierx package test suite
# ---------------------------------------------------------------------------
run_test() {
    log "Running hierx test suite"
    local hierx_root
    hierx_root="$(python -c 'import hierx, pathlib; print(pathlib.Path(hierx.__file__).parent.parent)')" 2>/dev/null || true

    if [[ -n "$hierx_root" && -d "${hierx_root}/tests" ]]; then
        python -m pytest --tb=short -q "${hierx_root}/tests/"
        echo "hierx tests passed."
    else
        echo "SKIP: hierx tests not found (pip install without source)."
        echo "  Install from source to run tests: pip install -e 'hierx[dev]'"
    fi

    # Paper pipeline smoke tests
    if [[ -d "${SCRIPT_DIR}/tests" ]]; then
        python -m pytest --tb=short -q "${SCRIPT_DIR}/tests/"
        echo "Pipeline smoke tests passed."
    fi
}

# ---------------------------------------------------------------------------
# figures — regenerate paper figures from pre-computed JSON (~10 sec)
# ---------------------------------------------------------------------------
run_figures() {
    log "Regenerating paper figures from pre-computed results"
    python -m benchmarks_final.plot_hierarchy_construction
    python -m benchmarks_final.plot_paper_figures
    python -m benchmarks_final.plot_baseline_comparison
    python -m benchmarks_final.plot_london_results

    # Figure 6 requires .npz accessibility arrays that are not tracked in git.
    # Try to generate it, but don't fail the whole run if data is missing.
    if [[ -f paper_figs_final/data/gb_drive_pop_accessibility_nodes.npz ]] && \
       [[ -f paper_figs_final/data/london_walk_1M_pop_accessibility_nodes.npz ]]; then
        python -m benchmarks_final.plot_accessibility_maps
    else
        echo ""
        echo "  SKIP: Figure 6 (accessibility maps) — .npz data files not found."
        echo "  Run './reproduce.sh data' to fetch from Zenodo, or"
        echo "  './reproduce.sh accessibility' to compute from scratch."
    fi

    echo ""
    echo "Figures written to paper_figs_final/figures/"
    ls -1 paper_figs_final/figures/*.pdf 2>/dev/null | wc -l | xargs -I{} echo "  {} PDF files generated"
    ls -1 paper_figs_final/figures/*.png 2>/dev/null | wc -l | xargs -I{} echo "  {} PNG files generated"
}

# ---------------------------------------------------------------------------
# benchmarks — rerun all synthetic benchmarks (~30-60 min)
# ---------------------------------------------------------------------------
run_benchmarks() {
    log "Running synthetic benchmarks"

    log "1/6  Scaling benchmark"
    python -m benchmarks_final.scaling_benchmark

    log "2/6  Error sensitivity"
    python -m benchmarks_final.error_sensitivity

    log "3/6  Baseline comparison (25k)"
    python -m benchmarks_final.baseline_comparison_25k

    log "4/6  Tuned scaling benchmark"
    python -m benchmarks_final.tuned_scaling_benchmark

    log "5/6  Layer structure benchmark"
    python -m benchmarks_final.layer_structure_benchmark

    log "6/6  Interaction function benchmark"
    python -m benchmarks_final.interaction_fn_benchmark

    echo ""
    echo "All synthetic benchmarks complete."
}

# ---------------------------------------------------------------------------
# london — fetch London OSM data + run London benchmarks (requires internet)
# ---------------------------------------------------------------------------
run_london() {
    # Try Zenodo first for archived network snapshots; fall back to OSM fetch
    if ! fetch_zenodo 2>/dev/null; then
        log "Zenodo unavailable — fetching London network data from OSM"
        python -m benchmarks_final.fetch_census_data
        python -m benchmarks_final.fetch_london_networks
        python -m benchmarks_final.fetch_london_walking
    fi

    log "Running London 60k benchmark"
    python -m benchmarks_final.london_benchmark_60k

    log "Running London walking 60k benchmark"
    python -m benchmarks_final.london_walk_benchmark_60k

    echo ""
    echo "London benchmarks complete."
}

# ---------------------------------------------------------------------------
# accessibility — compute accessibility maps (requires data mode first)
# ---------------------------------------------------------------------------
run_accessibility() {
    log "Computing accessibility maps"

    log "1/2  GB driving accessibility"
    run_pyrosm python -m benchmarks_final.run_uk_drive_accessibility

    log "2/2  London population accessibility"
    python -m benchmarks_final.run_population_accessibility

    echo ""
    echo "Accessibility computations complete."
}

# ---------------------------------------------------------------------------
# maps — generate interactive PMTiles maps
# ---------------------------------------------------------------------------
run_maps() {
    log "Generating interactive PMTiles maps"
    python interactive_maps/generate_pmtiles.py all
    echo ""
    echo "Maps written to interactive_maps/"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$MODE" in
    test)
        run_test
        ;;
    data)
        fetch_zenodo
        ;;
    data-fetch)
        run_data_fetch
        ;;
    figures)
        run_figures
        ;;
    benchmarks)
        run_benchmarks
        ;;
    london)
        run_london
        ;;
    accessibility)
        run_accessibility
        ;;
    maps)
        run_maps
        ;;
    reproduce)
        log "Full reproduction from Zenodo data"

        # Save pre-computed reference results before benchmarks overwrite them
        ref_dir="${SCRIPT_DIR}/.reproduce_reference"
        rm -rf "$ref_dir"
        mkdir -p "$ref_dir"
        cp "${SCRIPT_DIR}"/paper_figs_final/data/*.json "$ref_dir/" 2>/dev/null || true

        fetch_zenodo
        run_test
        run_benchmarks
        run_london
        run_accessibility
        run_figures

        log "Verifying reproduced results against reference"
        if ls "$ref_dir"/*.json &>/dev/null; then
            python -m benchmarks_final.compare_results \
                --reference "$ref_dir" \
                --reproduced paper_figs_final/data/
        else
            echo "  SKIP: No reference JSON files found to compare against."
        fi
        rm -rf "$ref_dir"
        ;;
    all)
        run_test
        run_data_fetch
        run_benchmarks
        run_london
        run_accessibility
        run_figures
        ;;
    *)
        echo "Usage: $0 {test|data|figures|benchmarks|london|accessibility|reproduce|data-fetch|maps|all}"
        echo ""
        echo "Modes:"
        echo "  test           Run hierx test suite"
        echo "  data           Fetch archived data from Zenodo"
        echo "  figures        Regenerate paper figures from pre-computed JSON (~10 sec)"
        echo "  benchmarks     Rerun all synthetic benchmarks (~30-60 min)"
        echo "  london         Run London network benchmarks (requires Zenodo data)"
        echo "  accessibility  Compute accessibility maps (requires Zenodo data, ~1-2 hours)"
        echo "  reproduce      Full reproduction from Zenodo: data → test → benchmarks → london → accessibility → figures → verify"
        echo "  data-fetch     Download all source data from origin (census + OSM networks)"
        echo "  maps           Generate interactive PMTiles maps (requires tippecanoe)"
        echo "  all            Full pipeline from source: data-fetch → test → benchmarks → london → accessibility → figures"
        exit 1
        ;;
esac
