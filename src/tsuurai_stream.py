#!/usr/bin/env python3
"""
TsuurAI Stream Mode Entry Point
Run with: streamlit run tsuurai_stream.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the stream app
from stream.app import *
