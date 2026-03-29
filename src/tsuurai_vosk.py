#!/usr/bin/env python3
"""
TsuurAI Vosk Real-time Entry Point
True streaming transcription with Vosk

Run with: streamlit run tsuurai_vosk.py
"""

import sys
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SRC_DIR))

# Run the vosk app with correct __file__ set
vosk_app_path = SRC_DIR / "stream" / "vosk_realtime.py"
vosk_globals = globals().copy()
vosk_globals['__file__'] = str(vosk_app_path)
exec(compile(open(vosk_app_path).read(), vosk_app_path, 'exec'), vosk_globals)
