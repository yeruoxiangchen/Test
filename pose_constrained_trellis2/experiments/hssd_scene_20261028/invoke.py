"""Explicit experiment-local decoder path mapping; no shared environment edits."""
import sys
from . import common as c
from pose_constrained_trellis2 import runtime

def main():
    m=c.models();runtime.SS_DECODER_MODEL=m['ss_decoder'];runtime.SHAPE_DECODER_MODEL=m['shape_decoder']
    action=sys.argv.pop(1)
    if action=='train':
        from pose_constrained_trellis2.train import main
    elif action=='infer':
        from pose_constrained_trellis2.infer import main
    else:raise ValueError(action)
    main()

if __name__=='__main__':main()
