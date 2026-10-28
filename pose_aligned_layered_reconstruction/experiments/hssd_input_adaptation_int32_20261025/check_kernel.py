"""GPU numerical regression: holey grids, shuffled rows, cross-tile neighbors."""
import json
import torch

from ...frozen_slat import configure_determinism, bootstrap_external_paths
from ..hssd_operator_audit_20261025.kernel_policy import FixedNative
from .tiled_native import LargeInputNative


@torch.no_grad()
def main():
    configure_determinism()
    bootstrap_external_paths()
    import spconv.pytorch as spconv
    from spconv.core import ConvAlgo
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    device = torch.device('cuda:0')
    grid = torch.cartesian_prod(torch.arange(17), torch.arange(16), torch.arange(15)).to(torch.int32)
    grid = grid[torch.rand(len(grid)) > .23]
    grid = grid[torch.randperm(len(grid))]
    coords = torch.cat([torch.zeros((len(grid), 1), dtype=torch.int32), grid], 1).to(device)
    rows = []
    for bias in (False, True):
        conv = spconv.SubMConv3d(192, 96, 3, bias=bias, algo=ConvAlgo.Native).half().to(device).eval()
        conv.weight.normal_(0, .04)
        if bias:
            conv.bias.normal_(0, .1)
        features = torch.randn((len(coords), 192), device=device, dtype=torch.float16) * .2
        def value():
            return spconv.SparseConvTensor(features, coords, [17, 16, 15], 1)
        with FixedNative():
            reference = conv(value())
        for tile in (257, 1024):
            with LargeInputNative(tile_rows=tile, force_signature=(192, 96)) as policy:
                actual = conv(value())
            torch.cuda.synchronize()
            record = {'bias': bias, 'tile_rows': tile, 'rows': len(coords),
                'coordinates_exact': torch.equal(reference.indices, actual.indices),
                'features_exact': torch.equal(reference.features, actual.features),
                'max_abs_error': float((reference.features - actual.features).abs().max()),
                'calls': policy.tiled_calls}
            rows.append(record)
            if not record['coordinates_exact'] or not record['features_exact']:
                raise AssertionError(json.dumps(record))
    from .common import REPAIR, HERE, write, now, binding
    write(REPAIR/'kernel_check.json', {'time': now(), 'passed': True, 'checks': rows,
        'sources': [binding(HERE/'tiled_native.py'), binding(__file__)]})
    print('KERNEL_REGRESSION_PASSED', flush=True)


if __name__ == '__main__':
    main()
