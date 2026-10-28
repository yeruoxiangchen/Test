import unittest

import numpy as np
import torch

from .metrics import binary_ranking,evaluate
from .scores import compute
from .visuals import voxelize


class LocalizationTests(unittest.TestCase):
    def test_ties_perfect_and_degenerate_rankings(self):
        y=np.array([1,0,1,0,0])
        r=binary_ranking(y,np.ones(5))
        self.assertAlmostEqual(r['ap'],.4);self.assertEqual(r['auroc'],.5)
        self.assertAlmostEqual(r['precision_top20'],.4)
        r=binary_ranking(y,y.astype(float))
        self.assertEqual(r['ap'],1.);self.assertEqual(r['auroc'],1.)
        self.assertFalse(binary_ranking(np.zeros(5),np.arange(5))['defined'])

    def test_same_camera_warp_and_spatial_shuffle(self):
        torch.set_num_threads(2)
        n=8;axis=(torch.arange(n)+.5)*518/n-.5
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        xyz=torch.stack(((xx-258.5)*2/400,(yy-258.5)*2/400,torch.full_like(xx,2.)),-1)
        pred=dict(xyz=xyz[None].repeat(2,1,1,1),depth=torch.full((2,n,n),2.),
                  mask=torch.ones((2,n,n),dtype=torch.bool),rgb=torch.zeros((2,n,n,3)))
        k=torch.tensor([[[400.,0,258.5],[0,400.,258.5],[0,0,1.]]]).repeat(2,1,1)
        w=torch.eye(4)[None].repeat(2,1,1)
        f=torch.randn((1,1369,1024),generator=torch.Generator().manual_seed(7)).repeat(2,1,1)
        rgb=np.random.default_rng(9).integers(0,255,(518,518,3),dtype=np.uint8)
        obs=np.ones((2,518,518),bool)
        a=compute(pred,[rgb,rgb],obs,f,f,k,k,w,0)
        self.assertTrue((a['support']==1).all())
        self.assertLess(float(a['cross_dino'].max()),2.e-6)
        self.assertLess(float(a['direct_dino'].max()),2.e-6)
        self.assertLess(float(a['cross_rgb'].max()),2.e-6)
        self.assertGreater(float(a['shuffled_dino'].mean()),.3)

    def test_voxelization_retains_unknown_and_out_of_bounds(self):
        xyz=np.array([[[[0,0,0],[.001,.001,.001],[.7,0,0]]]],np.float32)
        indices,scores,counts,outside=voxelize(xyz,np.array([[[.2,.8,1.]]]),np.ones((1,1,3),bool))
        self.assertEqual(outside,1);self.assertEqual(len(indices),1)
        self.assertAlmostEqual(float(scores[0]),.5);self.assertEqual(int(counts[0]),2)


if __name__=='__main__':unittest.main()
