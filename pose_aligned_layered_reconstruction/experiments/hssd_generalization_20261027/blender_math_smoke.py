"""Blender-only real API checks: instance transforms, ray casts, CV projection."""
import importlib.util
import json
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix,Vector
from bpy_extras.object_utils import world_to_camera_view

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_new_scene_capture_smoke',HERE/'scene_capture.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_plane_add(size=20);floor=bpy.context.object;floor.pass_index=1
bpy.context.view_layer.update();graph=bpy.context.evaluated_depsgraph_get()
hit,point,_,_,owner,_=bpy.context.scene.ray_cast(graph,m.C.to_3x3()@Vector([2,.3,2]),m.C.to_3x3()@Vector([0,-1,0]),distance=.8)
assert hit and owner.pass_index==1 and abs(point.z)<1e-6
vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
mesh=bpy.data.meshes.new('Prototype mesh');mesh.from_pydata([list(m.C.to_3x3()@Vector(v)) for v in vertices],[],[(0,1,2),(0,1,3)])
proto=bpy.data.objects.new('Prototype',mesh)
instance={'rotation':[.9238795325,0,.3826834324,0],'translation':[3,1,-2],'non_uniform_scale':[-2,.7,1.3]}
t=m.transform(instance);root,parts=m.h.instance([proto],'Transformed',m.C@t@m.CI,2)
bpy.context.view_layer.update()
actual=np.array([list(parts[0].matrix_world@v.co) for v in mesh.vertices]);expected=np.array([list(m.C@t@Vector([*v,1]))[:3] for v in vertices])
instance_error=float(np.max(np.abs(actual-expected)));assert instance_error<3e-5
scene=bpy.context.scene;side=2072;scene.render.resolution_x=scene.render.resolution_y=side;scene.render.resolution_percentage=100
cam=m.h.camera('Calibrated camera',[2,1.45,3],[0,.4,0]);scene.camera=cam
cam.data.sensor_fit='HORIZONTAL';cam.data.sensor_width=32
focal=.5*side/np.tan(np.deg2rad(70)/2);cam.data.lens=focal*32/side
bpy.context.view_layer.update()
pose=np.asarray(m.CI@cam.matrix_world@Matrix.Diagonal(Vector([1,-1,-1,1])),float);w=np.linalg.inv(pose)
points=np.array([[.2,.3,.1],[-.3,.7,-.2],[.1,1,-.3]],float);cv=np.c_[points,np.ones(len(points))]@w.T
uv=cv[:,:2]/cv[:,2:3]*focal+(side-1)/2
native=[]
for point in points:
    pixel=world_to_camera_view(scene,cam,m.C.to_3x3()@Vector(point));native.append([pixel.x*side-.5,(1-pixel.y)*side-.5])
projection_error=float(np.max(np.abs(uv-np.asarray(native))));assert projection_error<.003
assert np.max(np.abs(w[:3,:3]@w[:3,:3].T-np.eye(3)))<1e-6 and abs(np.linalg.det(w[:3,:3])-1)<1e-6
result={'passed':True,'real_blender_api':True,'floor_raycast':True,'negative_nonuniform_instance_error':instance_error,
        'world_cv_to_blender_projection_error_pixels':projection_error,'proper_camera_rotation':True,'cuda_model_loaded':False,
        'sources':[m.binding(Path(__file__)),m.binding(HERE/'scene_capture.py'),
                   m.binding(m.ROOT/'datasets/scene_occlusion_pilot_20261020/render_tools/render_hssd_blender.py')]}
out=m.OUT/'blender_math_smoke.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
print('BLENDER_MATH_SMOKE '+json.dumps(result),flush=True)
