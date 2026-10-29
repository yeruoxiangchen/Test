from .config import setup,snapshot
from .data import prepare


if __name__=='__main__':
    setup();prepare();snapshot();print('Frozen input bindings and diagnostic protocol prepared.',flush=True)
