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
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the batch app
from batch.app import *
