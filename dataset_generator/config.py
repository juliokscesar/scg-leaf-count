import sys
from pathlib import Path

_GN_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
def __include_packages():
    sys.path.append(_GN_ROOT_PATH)
