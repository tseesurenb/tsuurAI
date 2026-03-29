#!/usr/bin/env python3
"""
TsuurAI Stream Mode Entry Point
Run with: streamlit run tsuurai_stream.py
"""

import sys
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SRC_DIR))

# Run the stream app with correct __file__ set
stream_app_path = SRC_DIR / "stream" / "app.py"
stream_globals = globals().copy()
stream_globals['__file__'] = str(stream_app_path)
exec(compile(open(stream_app_path).read(), stream_app_path, 'exec'), stream_globals)
