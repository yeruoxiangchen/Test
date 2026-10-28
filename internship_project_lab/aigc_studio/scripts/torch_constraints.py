import importlib.metadata
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(''.join(f'{name}=={importlib.metadata.version(name)}\n' for name in ['torch', 'torchvision']))
