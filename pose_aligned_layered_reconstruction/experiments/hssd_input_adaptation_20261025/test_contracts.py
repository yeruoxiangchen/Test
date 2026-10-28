"""Projection/observation regression checks, independent of GPU inference."""
import unittest
import numpy as np
from PIL import Image
from .geometry import crop_affine,crop_image,resize_affine,projected_cube_box
from .common import *

class Contracts(unittest.TestCase):
    def test_pil_pixel_centers(self):
        # An integer pixel center in a cropped 2x resize lands midway between
        # the two corresponding destination pixel centers.
        box=[10,20,269,279];a=crop_affine(box)
        np.testing.assert_allclose(a@[10,20,1],[.5,.5,1],atol=1.e-12)
        np.testing.assert_allclose(a@[268,278,1],[516.5,516.5,1],atol=1.e-12)
        src=np.zeros((300,300),np.uint8);src[20,10]=255
        actual=np.asarray(crop_image(Image.fromarray(src),box,mask=True))
        expected=np.zeros((518,518),np.uint8);expected[:2,:2]=255
        np.testing.assert_array_equal(actual,expected)

    def test_high_and_low_same_final_camera(self):
        k=np.array([[471.,0,258.5],[0,471.,258.5],[0,0,1.]])
        for box in [[10,20,269,279],[0,0,518,518],[0,19,499,518]]:
            np.testing.assert_allclose(crop_affine(box)@k,crop_affine([4*x for x in box])@resize_affine(4)@k,atol=1.e-10)

    def test_border_crop_has_no_invented_observation(self):
        k=np.array([[400.,0,258.5],[0,400.,258.5],[0,0,1.]])
        w=np.eye(4);w[:3,3]=[-1.9,0,3]
        l,t,r,b=projected_cube_box(k,w)
        self.assertEqual(l,0);self.assertGreater(r-l,0);self.assertEqual(r-l,b-t)
        self.assertTrue(0<=t<b<=518 and 0<=l<r<=518)

    def test_frozen_real_views_and_reuse(self):
        if not (OUT/'plan.json').exists():self.skipTest('Run prepare for real-data checks')
        objects=plan()['plan']['objects']
        self.assertEqual([o['visible_indices'] for o in objects],[[0,1,2,3]]*3+[[0,3]])
        self.assertEqual(len(cases()),96);self.assertEqual(len(new_jobs()),54)
        for o in objects:
            for v in o['views']:
                mask=np.asarray(Image.open(verify(v['source_mask'])).convert('L'))>127
                y,x=np.nonzero(mask);l,t,r,b=v['box_518']
                self.assertTrue(((x>=l)&(x<r)&(y>=t)&(y<b)).all())
                self.assertTrue(0<=l<r<=518 and 0<=t<b<=518)

    def test_contexts_follow_selected_observations(self):
        if not (OUT/'plan.json').exists():self.skipTest('Run prepare for real-data checks')
        import torch
        from ...v2.dataset import select_views
        from ...dino_context import build_masked_dino_contexts
        obj=plan()['plan']['objects'][-1]
        source=torch.load(verify(obj['source_sample']),map_location='cpu',weights_only=True)
        result=select_views(source,torch.tensor([0,3]))
        for k in ['intrinsics','extrinsics','visible_target_mask','raw_visual_patch_features','masked_visual_patch_features']:
            self.assertTrue(torch.equal(result[k],source[k][[0,3]]),k)
        direct=build_masked_dino_contexts(source['masked_visual_patch_features'][[0,3]])
        self.assertTrue(torch.equal(result['stock_condition'],direct['stock_condition']))
        for k in direct['slat_condition']:
            self.assertEqual(len(result['slat_condition'][k]),len(direct['slat_condition'][k]))
            for a,b in zip(result['slat_condition'][k],direct['slat_condition'][k]):self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(source['target_coords'],result['target_coords']))

    def test_paired_summary_rejects_missing_case(self):
        from .summarize import paired,METRICS
        rows=[]
        for v in ['full_all','full_visible']:
            for seed in [42,43,44]:
                rows.append({'variant':v,'arm':'yxc30k','uid':'same_object','seed':seed,
                    'metrics':{k:.1 for k in METRICS},'reused_as_alias':v=='full_visible'})
        result=paired(rows,'yxc30k','full_all','full_visible')
        self.assertEqual(result['objects'],1);self.assertEqual(result['seed_pairs'],3)
        self.assertEqual(result['exact_input_aliases'],3)
        self.assertEqual(result['cd_wins'],0);self.assertEqual(result['fscore_wins'],0)
        with self.assertRaises(ValueError):paired(rows[:-1],'yxc30k','full_all','full_visible')

if __name__=='__main__':unittest.main()
