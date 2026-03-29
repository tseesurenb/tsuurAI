#!/bin/bash
# Run TsuurAI in Batch Mode
cd "$(dirname "$0")"
streamlit run tsuurai_batch.py --server.port 8501 --server.address 0.0.0.0 "$@"
