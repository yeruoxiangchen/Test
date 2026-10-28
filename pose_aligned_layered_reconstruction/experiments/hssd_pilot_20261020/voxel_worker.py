"""CPU-only official voxelizer; run in foundpose (Open3D .18 / NumPy 1.x)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if o3d.__version__ != '0.18.0' or int(np.__version__.split('.')[0]) != 1:
        raise RuntimeError('Unvalidated CPU voxelizer environment')
    mesh = o3d.io.read_triangle_mesh(a.mesh)
    mesh.vertices = o3d.utility.Vector3dVector(np.clip(np.asarray(mesh.vertices), -.5+1.e-6, .5-1.e-6))
    grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh,
        voxel_size=1/64, min_bound=(-.5, -.5, -.5), max_bound=(.5, .5, .5))
    coords = np.unique(np.array([v.grid_index for v in grid.get_voxels()], np.int32), axis=0)
    if not len(coords) or (coords < 0).any() or (coords >= 64).any():
        raise ValueError('Invalid official surface support')
    output = Path(a.output)
    if output.exists():
        with np.load(output) as old:
            if not np.array_equal(old['coords'], coords):
                raise ValueError('Existing surface support changed')
    else:
        np.savez_compressed(output, coords=coords)
    receipt = {'open3d': o3d.__version__, 'numpy': np.__version__, 'surface_count': len(coords),
        'mesh_sha256': hashlib.sha256(Path(a.mesh).read_bytes()).hexdigest(),
        'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'algorithm': 'official Open3D create_from_triangle_mesh_within_bounds 64 surface no fill'}
    Path(str(output)+'.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
