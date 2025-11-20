#!/bin/bash
# test_single_vortex.sh
#
# Simple single test - runs one evolution with full output visible.
# Use this to see what's actually happening and debug errors.

echo "========================================" 
echo "Single Vortex Test (w=1)"
echo "========================================" 
echo ""
echo "This will run ONE quantum evolution and show all output."
echo "If it fails, you'll see the error messages directly."
echo ""

# Very small system for testing
NX=3
NY=3
WINDING=1
T_MAX=2.0
DT=0.5

echo "Parameters:"
echo "  Lattice: ${NX}x${NY} (Hilbert space dim = 2^$(($NX*$NY)) = $((2**($NX*$NY))))"
echo "  Winding: ${WINDING}"
echo "  Time: ${T_MAX}"
echo "  Steps: $(echo "$T_MAX / $DT" | bc)"
echo ""
echo "Press Ctrl+C to abort, or wait 5 seconds to start..."
sleep 5

echo ""
echo "========================================" 
echo "Running evolution..."
echo "========================================" 
echo ""

python quantum_substrate_evolution.py \
    --Nx ${NX} \
    --Ny ${NY} \
    --pattern single_vortex \
    --winding ${WINDING} \
    --t_max ${T_MAX} \
    --dt ${DT} \
    --J_nn 1.0 \
    --J_defrag 0.5 \
    --mass 0.1 \
    --out test_output

EXIT_CODE=$?

echo ""
echo "========================================" 
echo "Test Complete"
echo "========================================" 
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS: Evolution completed"
    echo ""
    echo "Output directory: test_output/"
    
    if [ -f test_output/entanglement_evolution.json ]; then
        echo "Data file created: test_output/entanglement_evolution.json"
        echo ""
        echo "To analyze:"
        echo "  python analyze_entanglement_results.py test_output --plot"
    else
        echo "WARNING: Expected output file not found"
    fi
else
    echo "FAILED: Exit code ${EXIT_CODE}"
    echo ""
    echo "This means the quantum evolution failed."
    echo "Common causes:"
    echo "  - QuTiP not installed: pip install qutip"
    echo "  - Out of memory (try smaller NX, NY)"
    echo "  - Missing dependencies"
fi

echo ""