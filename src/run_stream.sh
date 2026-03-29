#!/bin/bash
# Run TsuurAI in Stream Mode
cd "$(dirname "$0")"
streamlit run tsuurai_stream.py --server.port 8502 --server.address 0.0.0.0 "$@"
