#!/bin/bash
# run_monogamy_tests.sh
#
# Automated test suite for monogamy of entanglement across different winding numbers.
# Tests the hypothesis: w=±1 patterns have monogamous entanglement, w=0 don't.

# Log everything
LOG_FILE="monogamy_tests.log"
exec 1> >(tee -a "${LOG_FILE}")
exec 2>&1

echo "========================================"
echo "Quantum Substrate: Monogamy Test Suite"
echo "========================================"
echo "Started: $(date)"
echo "Logging to: ${LOG_FILE}"
echo ""

# Configuration
LATTICE_SIZE=3  # Keep small for tractability
T_MAX=3.0
DT=0.2

# Test different winding numbers
WINDINGS=(0 1 -1)

echo "[INFO] Testing winding numbers: ${WINDINGS[@]}"
echo "[INFO] Lattice: ${LATTICE_SIZE}x${LATTICE_SIZE}"
echo "[INFO] Evolution time: ${T_MAX}"
echo ""

# Run evolution for each winding number
for w in "${WINDINGS[@]}"; do
    echo "========================================" 
    echo "Testing winding number w=${w}"
    echo "========================================" 
    
    OUT_DIR="results_w${w}"
    RUN_LOG="${OUT_DIR}_run.log"
    
    echo "[RUN] Evolving system with w=${w}..."
    echo "[RUN] Output logging to: ${RUN_LOG}"
    
    python quantum_substrate_evolution.py \
        --Nx ${LATTICE_SIZE} \
        --Ny ${LATTICE_SIZE} \
        --pattern single_vortex \
        --winding ${w} \
        --t_max ${T_MAX} \
        --dt ${DT} \
        --J_nn 1.0 \
        --J_defrag 0.5 \
        --mass 0.1 \
        --out ${OUT_DIR} 2>&1 | tee "${RUN_LOG}"
    
    EXIT_CODE=$?
    
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "[SUCCESS] w=${w} evolution complete"
    else
        echo "[ERROR] w=${w} evolution failed with exit code ${EXIT_CODE}"
        echo "[ERROR] Check ${RUN_LOG} for details"
    fi
    
    echo ""
done

echo "========================================"
echo "Analyzing Results"
echo "========================================"
echo ""

# Analyze each result
for w in "${WINDINGS[@]}"; do
    OUT_DIR="results_w${w}"
    
    if [ -d "${OUT_DIR}" ] && [ -f "${OUT_DIR}/entanglement_evolution.json" ]; then
        echo "--- Analyzing winding w=${w} ---"
        python analyze_entanglement_results.py ${OUT_DIR} 2>&1 | tee "${OUT_DIR}_analysis.log"
    else
        echo "[SKIP] No results found for w=${w}"
    fi
    echo ""
done

# Generate comparison plot
echo "========================================" 
echo "Generating Comparison Plot"
echo "========================================" 
echo ""

RESULT_DIRS=()
VALID_WINDINGS=()
for w in "${WINDINGS[@]}"; do
    if [ -f "results_w${w}/entanglement_evolution.json" ]; then
        RESULT_DIRS+=("results_w${w}")
        VALID_WINDINGS+=("${w}")
    fi
done

if [ ${#RESULT_DIRS[@]} -gt 0 ]; then
    echo "[INFO] Comparing ${#RESULT_DIRS[@]} results: ${VALID_WINDINGS[@]}"
    
    python analyze_entanglement_results.py dummy \
        --compare "${RESULT_DIRS[@]}" \
        --windings "${VALID_WINDINGS[@]}" \
        --output monogamy_comparison.png 2>&1 | tee comparison_plot.log

    if [ -f monogamy_comparison.png ]; then
        echo "[SUCCESS] Comparison plot saved: monogamy_comparison.png"
    else
        echo "[WARNING] Comparison plot not generated"
        echo "[WARNING] Check comparison_plot.log for errors"
    fi
else
    echo "[ERROR] No valid results to compare"
fi

echo ""
echo "========================================"
echo "Test Suite Complete"
echo "========================================"
echo "Completed: $(date)"
echo ""
echo "Results:"
for w in "${WINDINGS[@]}"; do
    if [ -d "results_w${w}" ]; then
        echo "  w=${w}: results_w${w}/ [EXISTS]"
    else
        echo "  w=${w}: results_w${w}/ [MISSING]"
    fi
done
echo ""
if [ -f monogamy_comparison.png ]; then
    echo "Comparison plot: monogamy_comparison.png [EXISTS]"
else
    echo "Comparison plot: monogamy_comparison.png [MISSING]"
fi
echo ""
echo "Logs:"
echo "  Main log: ${LOG_FILE}"
for w in "${WINDINGS[@]}"; do
    if [ -f "results_w${w}_run.log" ]; then
        echo "  w=${w} run: results_w${w}_run.log"
    fi
    if [ -f "results_w${w}_analysis.log" ]; then
        echo "  w=${w} analysis: results_w${w}_analysis.log"
    fi
done
if [ -f comparison_plot.log ]; then
    echo "  Comparison: comparison_plot.log"
fi
echo ""
echo "To view errors, check the log files above."
echo ""