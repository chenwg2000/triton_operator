import triton
import triton.language as tl
import torch

print(f'Triton version: {triton.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Compute capability: {torch.cuda.get_device_capability(0)}')

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

size = 1024
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output = torch.empty(size, device='cuda')
grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, output, size, BLOCK_SIZE=256)
expected = x + y
print(f'Max error: {(output - expected).abs().max().item()}')
print('Triton kernel executed successfully on SM120!')
