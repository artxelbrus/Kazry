import torch
import torch.nn as nn
import triton
import triton.language as tl
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def kazry_forward(
    x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr, N):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr+offs, mask=mask).to(tl.float32)
    exp = tl.exp(x)
    scale = tl.where(x >= 0.0, 1.0, exp)
    x = x * scale
    tl.store(y_ptr+offs, x.to(tl.bfloat16), mask=mask)
@triton.jit
def kazry_forward_in_place(
    x_ptr, BLOCK_SIZE: tl.constexpr, N):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr+offs, mask=mask).to(tl.float32)
    exp = tl.exp(x)
    scale = tl.where(x >= 0.0, 1.0, exp)
    x = x * scale
    tl.store(x_ptr+offs, x.to(tl.bfloat16), mask=mask)
@triton.jit
def kazry_backward(
    x_ptr, grad_ptr, grad_out_ptr, BLOCK_SIZE: tl.constexpr, N):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr+offs, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr+offs, mask=mask).to(tl.float32)
    grad = grad * tl.where(x >= 0.0, 1.0, tl.exp(x) * (1 + x))
    tl.store(grad_out_ptr+offs, grad.to(tl.bfloat16), mask=mask)
class KazryTritonWithBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        BLOCK_SIZE=512
        N = x.numel()
        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(N meta['BLOCK_SIZE']),)
        kazry_forward[grid](x_ptr=x, y_ptr=y, BLOCK_SIZE=BLOCK_SIZE, N=N)
        ctx.save_for_backward(x)
        return y
    @staticmethod
    def backward(ctx, grad):
        BLOCK_SIZE=1024
        x = ctx.saved_tensors[0]
        N = grad.numel()
        grad_out = torch.empty_like(grad)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        kazry_backward[grid](x_ptr=x, grad_ptr=grad, grad_out_ptr=grad_out, BLOCK_SIZE=BLOCK_SIZE, N=N)
        return grad_out
class KazryTritonInPlace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        BLOCK_SIZE=512
        N = x.numel()
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        kazry_forward_in_place[grid](x_ptr=x, BLOCK_SIZE=BLOCK_SIZE, N=N)
        return x
    @staticmethod
    def backward(ctx, grad):
        raise RuntimeError("Err, Kazry inplace does not support backward, use kazry with backward")
        return None
class KazryWithBackward(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return KazryTritonWithBackward.apply(x)
class KazryInPlace(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return KazryTritonInPlace.apply(x)
def kazry_with_backward(x):
    return KazryTritonWithBackward.apply(x)
def kazry_inplace(x):
    return KazryTritonInPlace.apply(x)
