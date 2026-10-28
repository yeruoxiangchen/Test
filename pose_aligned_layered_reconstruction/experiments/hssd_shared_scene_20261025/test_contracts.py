"""Small geometry-contract tests for the new shared-world interface."""
import unittest
import numpy as np
from .geometry import trajectory, canonical_extrinsics, project, C


class GeometryContracts(unittest.TestCase):
    def test_uniform_scale_does_not_change_projection(self):
        _,k,w=trajectory()
        a=np.eye(4)
        a[:3,:3]=.37*C[:3,:3]
        a[:3,3]=[-.8,.3,-.1]
        local=canonical_extrinsics(w,a)
        xyz=np.array([[.1,.2,.3],[-.2,.1,.1],[0.,0.,0.]])
        world=(np.c_[xyz,np.ones(len(xyz))]@np.linalg.inv(a).T)[:,:3]
        for kk,ww,ll in zip(k,w,local):
            np.testing.assert_allclose(project(world,kk,ww)[0],project(xyz,kk,ll)[0],atol=1.e-8)
            np.testing.assert_allclose(project(xyz,kk,ll)[1],.37*project(world,kk,ww)[1],atol=1.e-8)
            np.testing.assert_allclose(ll[:3,:3]@ll[:3,:3].T,np.eye(3),atol=1.e-8)

    def test_nonuniform_object_frame_is_rejected(self):
        _,_,w=trajectory()
        with self.assertRaises(ValueError):
            canonical_extrinsics(w,np.diag([1.,2.,1.,1.]))

    def test_world_target_projects_to_center_in_all_frames(self):
        eyes,k,w=trajectory()
        self.assertEqual(len(eyes),12)
        for kk,ww in zip(k,w):
            uv,z=project([[2.7,.55,-1.5]],kk,ww)
            np.testing.assert_allclose(uv,[[258.5,258.5]],atol=1.e-8)
            self.assertGreater(z[0],0)
        self.assertLess(np.linalg.norm(np.diff(eyes,axis=0),axis=1).max(),.5)

    def test_blender_pose_conversion_preserves_forward_ray(self):
        _,_,w=trajectory()
        cv_to_gl=np.diag([1.,-1.,-1.,1.])
        for ww in w:
            blender_c2w=C@np.linalg.inv(ww)@cv_to_gl
            cv_forward=np.linalg.inv(ww)@np.array([0.,0.,1.,1.])
            blender_forward=blender_c2w@np.array([0.,0.,-1.,1.])
            np.testing.assert_allclose(blender_forward,C@cv_forward,atol=1.e-8)


if __name__=='__main__':unittest.main()
