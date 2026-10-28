import copy
import unittest

import numpy as np
import torch

from .prepare import select_rows
from .report import stats, strata
from .evaluate import error_regions, score_error_regions


class ProtocolTests(unittest.TestCase):
    def test_cohort_selection_order_and_exclusion(self):
        objects=[dict(uid=f'{c}/{i}',category=c) for c in 'abcd' for i in range(4)]
        excluded={f'{c}/0' for c in 'abcd'}
        a=select_rows(objects,excluded,count=3)
        b=select_rows(list(reversed(objects)),excluded,count=3)
        self.assertEqual(a,b)
        self.assertFalse({r['uid'] for r in a}&excluded)
        self.assertEqual(len({r['category'] for r in a}),3)

    def test_object_statistics_do_not_treat_seeds_as_independent(self):
        value=stats([('a',1.),('a',3.),('b',6.)])
        self.assertEqual(value['mean'],4.)
        self.assertEqual(value['objects'],2)
        self.assertEqual(value['cases'],3)

    def test_strata_use_base_not_candidate(self):
        rows=[dict(uid=u,original_metrics={'surface':{'chamfer_l1':v}},metrics={'accepted':{'value':10-v}})
              for u,v in [('a',1),('b',3),('c',2),('d',4)]]
        before=strata(rows,'original_metrics.surface.chamfer_l1')
        for r in rows:r['metrics']['accepted']['value']=-100
        self.assertEqual(before,strata(rows,'original_metrics.surface.chamfer_l1'))
        self.assertEqual(set(before['higher']),{'b','d'})

    def test_disappearing_rays_cannot_remove_bad_region(self):
        target=dict(depth=torch.tensor([[[1.,1.]]]),mask=torch.ones((1,1,2),dtype=torch.bool))
        base=dict(depth=torch.tensor([[[1.02,1.]]]),mask=target['mask'].clone())
        masks=error_regions(base,target,torch.ones((1,1,2)))
        old=score_error_regions(base,target,masks)
        edited=dict(depth=torch.tensor([[[0.,1.]]]),mask=torch.tensor([[[False,True]]]))
        new=score_error_regions(edited,target,masks)
        self.assertEqual(new['bad_pixels'],1)
        self.assertGreater(new['bad_depth_mae'],old['bad_depth_mae'])
        self.assertEqual(new['good_depth_mae'],0.)
        self.assertAlmostEqual(new['selection_precision'],.5)

    def test_engine_binding_keeps_old_files_out_of_scope(self):
        from .config import OUT,bind_engine,folder
        engine=bind_engine()
        self.assertEqual(engine.OUT,OUT)
        self.assertEqual(engine.folder(100,42),folder(100,42))
        self.assertEqual(engine.data_folder(100),OUT/'data/100')
        self.assertEqual(engine.PARAMS['dev_indices'],[])


if __name__=='__main__':unittest.main()
