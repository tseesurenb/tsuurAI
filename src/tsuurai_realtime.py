#!/usr/bin/env python3
"""
TsuurAI Real-time Mode Entry Point
True real-time transcription using WebRTC

Run with: streamlit run tsuurai_realtime.py
"""

import sys
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SRC_DIR))

# Run the realtime app with correct __file__ set
realtime_app_path = SRC_DIR / "stream" / "realtime_app.py"
realtime_globals = globals().copy()
realtime_globals['__file__'] = str(realtime_app_path)
exec(compile(open(realtime_app_path).read(), realtime_app_path, 'exec'), realtime_globals)
