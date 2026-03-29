#!/usr/bin/env python3
"""
TsuurAI Batch Mode Entry Point
Run with: streamlit run tsuurai_batch.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the batch app
from batch.app import *
