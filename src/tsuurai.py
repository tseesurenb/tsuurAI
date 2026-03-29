#!/usr/bin/env python3
"""
TsuurAI Main Entry Point (Batch Mode)
For backward compatibility - runs batch mode by default

Usage:
    Batch mode: streamlit run tsuurai.py (or tsuurai_batch.py)
    Stream mode: streamlit run tsuurai_stream.py
"""

import sys
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SRC_DIR))

# Run the batch app with correct __file__ set
batch_app_path = SRC_DIR / "batch" / "app.py"
batch_globals = globals().copy()
batch_globals['__file__'] = str(batch_app_path)
exec(compile(open(batch_app_path).read(), batch_app_path, 'exec'), batch_globals)
