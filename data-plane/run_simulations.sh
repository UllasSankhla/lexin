#!/usr/bin/env bash
# Run all simulation and unit tests, writing output to logs/simulations_<timestamp>.log
# Usage: ./run_simulations.sh [--no-timestamp]  (--no-timestamp writes to logs/simulations_latest.log)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

if [[ "${1:-}" == "--no-timestamp" ]]; then
    LOG_FILE="$LOG_DIR/simulations_latest.log"
else
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="$LOG_DIR/simulations_${TIMESTAMP}.log"
    # Also update the "latest" symlink
    ln -sf "$LOG_FILE" "$LOG_DIR/simulations_latest.log"
fi

echo "Running simulations → $LOG_FILE"

{
    echo "========================================"
    echo "Simulation run: $(date)"
    echo "Model: ${LLM_PROVIDER:-cerebras} / ${CEREBRAS_MODEL:-default}"
    echo "========================================"
    echo ""

    cd "$SCRIPT_DIR"

    SUITES=(
        "tests/simulate_data_collection.py"
        "tests/simulate_faq_interrupt.py"
        "tests/simulate_farewell.py"
        "tests/simulate_fallback.py"
        "tests/simulate_intake_qualification.py"
        "tests/simulate_narrative_embedded_questions.py"
        "tests/test_narrative_collection.py"
        "tests/test_tools.py"
        "tests/test_intake_qualification.py"
        "tests/test_scheduling_flow.py"
    )

    PASS=0
    FAIL=0
    ERROR=0

    for SUITE in "${SUITES[@]}"; do
        echo "----------------------------------------"
        echo "SUITE: $SUITE"
        echo "----------------------------------------"
        if PYTHONPATH=. python3 -m pytest "$SUITE" -v --tb=short 2>&1; then
            PASS=$((PASS + 1))
        else
            FAIL=$((FAIL + 1))
        fi
        echo ""
    done

    echo "========================================"
    echo "SUMMARY"
    echo "========================================"
    echo "Suites passed : $PASS"
    echo "Suites failed : $FAIL"
    echo "Run complete  : $(date)"
    echo "Log file      : $LOG_FILE"

} 2>&1 | tee "$LOG_FILE"
