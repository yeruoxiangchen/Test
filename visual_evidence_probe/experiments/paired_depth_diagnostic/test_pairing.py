import unittest
import numpy as np
from .prepare import matched_support


class PairingTests(unittest.TestCase):
    def test_both_sources_use_positive_finite_gt_intersection_and_no_heldout(self):
        valid=np.ones((3,2,2),bool);gt=np.ones((3,2,2));mask=np.ones_like(valid);region=np.ones_like(gt)
        gt[0,0,0]=0;gt[0,0,1]=float('nan');mask[1,0,0]=False
        result=matched_support(valid,gt,mask,region,[0,1],[2])
        self.assertEqual(int(result.sum()),5);self.assertFalse(result[2].any())
        self.assertTrue(valid.all());self.assertTrue(np.array_equal(result,matched_support(valid,gt,mask,region,[0,1],[2])))

    def test_input_region_exclusion_is_shared(self):
        valid=np.ones((2,2,2),bool);gt=np.ones_like(valid,dtype=np.float32);region=np.ones_like(gt);region[0,0,0]=0
        result=matched_support(valid,gt,valid,region,[0],[1]);self.assertEqual(int(result.sum()),3)

    def test_overlap_is_rejected(self):
        valid=np.ones((2,2,2),bool)
        with self.assertRaises(ValueError):matched_support(valid,valid.astype(float),valid,valid,[0,1],[1])


if __name__=='__main__':unittest.main()
