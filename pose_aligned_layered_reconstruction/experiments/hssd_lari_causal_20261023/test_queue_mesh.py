import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import trimesh
from . import mesh_cases, runner
from .common import binding, read, write


class QueueMeshTests(unittest.TestCase):
    def test_queue_has_only_registered_inference_stages(self):
        with tempfile.TemporaryDirectory() as folder:
            calls=[]
            with patch.object(runner,'OUT',Path(folder)), patch.object(runner,'require_preflight'), patch.object(runner,'child',side_effect=lambda stage,command:calls.append((stage,command))):
                runner.queue('run')
            self.assertEqual(len(calls),10)
            self.assertEqual([s for s,_ in calls],['ss_predicted_point','ss_oracle_point','ss_summary',
                'mesh_base','mesh_appearance_control','mesh_oracle_point','mesh_oracle_cell','mesh_predicted_point','mesh_predicted_cell','mesh_summary'])
            self.assertTrue(all('train' not in ' '.join(command) for _,command in calls))
            self.assertEqual(read(Path(folder)/'status.json')['status'],'diagnostics_complete_review_required')

    def test_mesh_uses_global_index_native_frame_and_completed_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);out=base/'out';cell=base/'cell'
            original=cell/'evaluations'/'occluded_base'
            write(original/'identity.json',{'test':True});write(original/'ss_complete.json',{'test':True})
            gt=base/'source_gt.ply';mesh=trimesh.creation.box(extents=[.5,.4,.3]);mesh.export(gt)
            gtb=binding(gt);uid='object_global_2:occluded'
            source=original/'002';source.mkdir()
            np.savez_compressed(source/'42_ss.npz',coords=np.array([[0,1,2,3]],np.int32))
            write(source/'42_ss.json',{'uid':uid,'object_uid':'object_global_2','index':2,'seed':42,'arm':'base',
                'gt':gtb,'coords':binding(source/'42_ss.npz'),'success':True})
            samples=[None,None,{'uid':uid,'object_uid':'object_global_2','target_mesh_binding':gtb}]
            routed=[]
            class Decoded:
                def to_trimesh(self,*,transform_pose):
                    routed.append(('transform_pose',transform_pose));return mesh.copy()
            runtime={'binding':{'test':True},'decoder':lambda z:[Decoded()]}
            def noise(seed,index,device):routed.append(('noise',seed,index));return None
            patches=[patch.object(mesh_cases,'OUT',out),patch.object(mesh_cases,'CELL',cell),
                patch.object(mesh_cases,'INDICES',(2,)),patch.object(mesh_cases,'SEEDS',(42,)),
                patch.object(mesh_cases,'freeze',return_value={'test':True}),patch.object(mesh_cases,'SceneDataset',return_value=samples),
                patch.object(mesh_cases,'select_views',side_effect=lambda s,i:s),patch.object(mesh_cases,'install_memory_efficient_mesh'),
                patch.object(mesh_cases,'load_frozen_slat30k',return_value=runtime),patch.object(mesh_cases,'master_noise',side_effect=noise),
                patch.object(mesh_cases,'sparse_noise',return_value=None),patch.object(mesh_cases,'sample_mesh_latent',return_value=(None,None)),
                patch.object(mesh_cases,'surface_metrics',return_value={'chamfer_l1':0.}),patch.object(mesh_cases,'structure_metrics',return_value={'mesh_success':True})]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for p in patches:stack.enter_context(p)
                mesh_cases.decode('base')
                mesh_cases.decode('base')
            self.assertEqual(routed,[('noise',42,2),('transform_pose',False)])
            row=read(out/'mesh'/'base'/'002'/'42.json')
            self.assertTrue(row['success']);self.assertFalse(row['fitted_transform'])
            self.assertEqual(row['gt']['sha256'],gtb['sha256'])
            self.assertEqual(row['native_bbox_min'],row['gt_bbox_min'])


if __name__=='__main__':unittest.main()
