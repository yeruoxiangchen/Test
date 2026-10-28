import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from . import cropped_probe as probe


class CroppedPairingTests(unittest.TestCase):
    def test_mixed_images_keep_their_own_intrinsics(self):
        features = torch.arange(8).float()[:,None,None].expand(8,4,1024).clone()
        k = torch.eye(3).repeat(8,1,1)
        obs = dict(intrinsics=k, extrinsics=torch.eye(4).repeat(8,1,1), image_size=(518,518),
                   grid_transform='identity', extrinsics_type='w2c', camera_forward_sign=1.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({'features':features,'intrinsics':k},root/'original_crop.pt')
            cache = {}
            for key,offset in [('a',100),('b',200),('gt',300)]:
                intrinsics = k.clone()
                intrinsics[:,0,0] = torch.arange(8)+offset
                cache[key]={'features':features+offset,'intrinsics':intrinsics}
            torch.save(cache,root/'render_features.pt')
            with patch.object(probe,'folder',return_value=root), patch.object(probe.data,'object_data',return_value=({},obs)):
                ab=probe.input_for(0,'mixed_ab6'); ba=probe.input_for(0,'mixed_ba6')
            for j in range(2,8):
                for sample,offset in [(ab,100 if j%2==0 else 200),(ba,200 if j%2==0 else 100)]:
                    self.assertEqual(float(sample['masked_visual_patch_features'][j,0,0]),j+offset)
                    self.assertEqual(float(sample['intrinsics'][j,0,0]),j+offset)
            self.assertTrue(torch.equal(ab['extrinsics'],ba['extrinsics']))
            self.assertNotIn('target_mesh_binding',ab)


if __name__=='__main__': unittest.main()
