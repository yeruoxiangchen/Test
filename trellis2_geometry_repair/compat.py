"""Process-local FlexGEMM fix for frozen weights; forward stays upstream."""
import importlib


def install_frozen_weight_backward():
    module=importlib.import_module('flex_gemm.ops.spconv.submanifold_conv3d')
    cls=module.SubMConv3dFunction
    if getattr(cls,'_t2_repair_frozen_fix',False):return
    original=cls._sparse_submanifold_conv_backward

    def backward_values(grad_output,feats,cache,weight,bias=None):
        if weight.requires_grad or module.spconv.ALGORITHM==module.Algorithm.EXPLICIT_GEMM:
            return original(grad_output,feats,cache,weight,bias)
        algorithms={
            module.Algorithm.IMPLICIT_GEMM:'sparse_submanifold_conv_bwd_implicit_gemm',
            module.Algorithm.IMPLICIT_GEMM_SPLITK:'sparse_submanifold_conv_bwd_implicit_gemm_splitk',
            module.Algorithm.MASKED_IMPLICIT_GEMM:'sparse_submanifold_conv_bwd_masked_implicit_gemm',
            module.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK:'sparse_submanifold_conv_bwd_masked_implicit_gemm_splitk'}
        algorithm=module.spconv.ALGORITHM
        if algorithm not in algorithms:raise RuntimeError(f'unsupported frozen backward {algorithm}')
        co,kw,kh,kd,ci=weight.shape
        args=[grad_output.contiguous(),feats,weight.reshape(co,kd*kh*kw,ci),bias,cache['neighbor_map']]
        if algorithm in (module.Algorithm.MASKED_IMPLICIT_GEMM,module.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK):
            args.extend(cache[key] for key in ('sorted_idx','valid_kernel_callback','valid_kernel_seg_callback',
                                               'valid_signal_i','valid_signal_o','valid_signal_seg'))
        grad_input,grad_weight,grad_bias=getattr(module.kernels.triton,algorithms[algorithm])(*args)
        # Same upstream kernel and input gradient. A frozen weight correctly
        # returns None; upstream incorrectly calls None.reshape afterwards.
        if grad_weight is not None:grad_weight=grad_weight.reshape(co,kw,kh,kd,ci)
        return grad_input,grad_weight,grad_bias

    def backward(ctx,grad_output,_):
        feats,weight,bias=ctx.saved_tensors
        gi,gw,gb=backward_values(grad_output,feats,ctx.neighbor_cache,weight,bias)
        return (gi if feats.requires_grad else None,None,None,None,
                gw if weight.requires_grad else None,
                gb if bias is not None and bias.requires_grad else None,None)

    cls._sparse_submanifold_conv_backward=staticmethod(backward_values)
    cls.backward=staticmethod(backward)
    cls._t2_repair_frozen_fix=True


def check_cuda():
    import torch
    from .config import OUT,write,now,setup
    setup();install_frozen_weight_backward()
    from flex_gemm.ops.spconv import sparse_submanifold_conv3d
    torch.manual_seed(81)
    coords=torch.cartesian_prod(*[torch.arange(3,device='cuda:0',dtype=torch.int32)]*3)
    coords=torch.cat([torch.zeros((len(coords),1),device='cuda:0',dtype=torch.int32),coords],1)
    features=(torch.randn((len(coords),8),device='cuda:0')*.1).half()
    weights=(torch.randn((4,3,3,3,8),device='cuda:0')*.1).half()
    bias=torch.zeros(4,device='cuda:0',dtype=torch.float16)
    results=[]
    for train_weights in (True,False):
        x=features.detach().clone().requires_grad_(True);w=weights.detach().clone().requires_grad_(train_weights)
        b=bias.detach().clone().requires_grad_(train_weights)
        output,_=sparse_submanifold_conv3d(x,coords,torch.Size([1,8,3,3,3]),weight=w,bias=b)
        output.float().square().sum().backward()
        results.append((output.detach(),x.grad.detach()))
    forward=float((results[0][0]-results[1][0]).abs().max())
    grad=float((results[0][1]-results[1][1]).abs().max())
    passed=forward==0 and bool(torch.allclose(results[0][1],results[1][1],atol=.002,rtol=.005))
    value=dict(time=now(),passed=passed,forward_max_error=forward,input_gradient_max_error=grad,
               upstream_kernel_unchanged=True,site_packages_files_modified=False)
    write(OUT/'validation/flex_gemm_frozen.json',value);print(value,flush=True)
    if not passed:raise RuntimeError('frozen-weight gradient compatibility failed')


if __name__=='__main__':check_cuda()
