from pathlib import Path
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
_src = Path(__file__).resolve().parent / 'src'
if _src.is_dir():
    __path__.append(str(_src))
