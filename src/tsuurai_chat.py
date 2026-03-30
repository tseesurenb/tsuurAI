"""
TsuurAI Chat Mode
Launch with: streamlit run tsuurai_chat.py
"""

import runpy
import sys
from pathlib import Path

# Ensure correct __file__ for Streamlit
sys.modules[__name__].__file__ = str(Path(__file__).parent / "chat" / "app.py")
runpy.run_path(str(Path(__file__).parent / "chat" / "app.py"), run_name="__main__")
