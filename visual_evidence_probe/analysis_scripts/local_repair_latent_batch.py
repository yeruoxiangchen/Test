"""Reuse immutable GT/evaluation caches across arms; numerical objectives unchanged."""
import functools
import hashlib
from pathlib import Path

from . import local_repair_latent as original
from . import local_repair_latent_run as safe
from . import local_repair_latent_deterministic as deterministic


def main():
    from visual_evidence_probe.experiments.local_repair import runner
    runner.load_gt = functools.lru_cache(maxsize=1)(runner.load_gt)
    original_evaluator = runner.make_evaluator
    evaluators = {}
    def evaluator(gt, pred_base, target_eval, masks):
        key = (id(gt), hashlib.sha256(pred_base['depth'].cpu().numpy().tobytes()).hexdigest())
        if key not in evaluators:
            evaluators.clear()
            evaluators[key] = original_evaluator(gt, pred_base, target_eval, masks)
        return evaluators[key]
    runner.make_evaluator = evaluator
    old = safe.base_register
    def register():
        old()
        receipt = dict(format='local_slat_batch_cache.v1', source=dict(path=str(Path(__file__).resolve()),
            sha256=original.config.digest(__file__)),
            change='LRU cache one immutable GT mesh and reuse exactly identical fixed-domain evaluator across arms.',
            numerical_objectives_changed=False)
        p = original.OUT / 'batch_cache_amendment.json'
        if p.exists() and original.config.read(p) != receipt:
            raise ValueError('cache amendment changed')
        original.config.write(p, receipt)
    safe.base_register = register
    safe.main()


if __name__ == '__main__': main()
