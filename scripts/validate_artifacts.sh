#!/usr/bin/env bash
# Wrapper that runs the Tcl validator on every grok-4.3 artifact.
# Workaround for Vivado wrapper's cmd.exe quoting bug: keep the .tcl
# in the wrapper's Windows CWD (C:\Users\Giorgos) so the -source path
# is a single token without backslashes, sidestepping the double-quote
# corruption.
set -u
cd "$(dirname "$0")/.."

VIVADO="${VIVADO_EXEC:-/home/giorgos/.local/bin/vivado}"

# The Tcl source path must avoid the Vivado WSL2 wrapper's cmd.exe
# quoting bug: backslash-containing -source paths get double-quoted
# and Vivado treats the quotes as part of the filename. Workaround:
# stage the Tcl in the wrapper's Windows CWD (C:\Users\Giorgos) so
# the relative path passed to -source is a single token. Auto-staged
# here if missing or stale.
TCL_STAGED="/mnt/c/Users/Giorgos/_validate_artifacts.tcl"
if [[ ! -f "$TCL_STAGED" ]] || [[ "scripts/validate_artifacts.tcl" -nt "$TCL_STAGED" ]]; then
    cp "scripts/validate_artifacts.tcl" "$TCL_STAGED"
fi

ARTIFACTS=(
    "live_tests/_steer_ispd16/ispd16_example2_optimized.dcp"
    "live_tests/_steer_finn/finn_radioml_optimized.dcp"
    "live_tests/_steer_finn_rep2/finn_radioml_optimized.dcp"
    "live_tests/_steer_corescore/corescore_500_mod_optimized.dcp"
)

for dcp in "${ARTIFACTS[@]}"; do
    if [[ ! -f "$dcp" ]]; then
        echo "VALIDATE-WRAPPER:SKIP:$dcp (file not found)"
        continue
    fi
    saved_edif="${dcp%.dcp}.edf"
    log="${dcp}.verify.log"

    echo "=== $dcp ==="
    echo "  DCP size: $(stat -c %s "$dcp") bytes"
    if [[ -f "$saved_edif" ]]; then
        echo "  Saved EDIF size: $(stat -c %s "$saved_edif") bytes"
    fi
    t0=$(date +%s)
    "$VIVADO" -mode batch -nojournal -nolog -source _validate_artifacts.tcl -tclargs "$dcp" \
        > "$log" 2>&1
    rc=$?
    t1=$(date +%s)
    echo "  Vivado batch elapsed: $((t1 - t0))s, exit=$rc"
    grep -E "^VALIDATE:" "$log" | grep -v "ROUTE_STATUS_BLOCK" | head -10
    fresh="${dcp}.verify.edif"
    if [[ -f "$fresh" && -f "$saved_edif" ]]; then
        fresh_size=$(stat -c %s "$fresh")
        saved_size=$(stat -c %s "$saved_edif")
        delta=$((fresh_size - saved_size))
        echo "  Fresh EDIF size: $fresh_size bytes (saved Δ = $delta bytes)"
    fi
    echo
done
