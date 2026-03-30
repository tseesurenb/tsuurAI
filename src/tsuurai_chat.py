#!/usr/bin/env python3
"""
TsuurAI Chat Mode Entry Point
Run with: streamlit run tsuurai_chat.py
"""

import sys
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SRC_DIR))

# Run the chat app with correct __file__ set
chat_app_path = SRC_DIR / "chat" / "app.py"
chat_globals = globals().copy()
chat_globals['__file__'] = str(chat_app_path)
exec(compile(open(chat_app_path).read(), chat_app_path, 'exec'), chat_globals)
