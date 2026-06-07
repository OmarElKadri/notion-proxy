"""Make the repo root importable so `import notion_proxy` works without install."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
