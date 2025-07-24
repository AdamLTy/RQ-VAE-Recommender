import paddle
import triton
import triton.language as tl

from paddle import Tensor
from paddle.autograd import PyLayer
# TODO: Handle nested tensor equivalent in Paddle
# from paddle.nested import Tensor as NestedTensor
NestedTensor = Tensor  # Temporary fallback


class PaddedToJaggedTensor(PyLayer):
    @staticmethod
    def forward(ctx, x: Tensor, lengths: Tensor, max_len: int) -> NestedTensor:
        assert x.dim() == 3
        assert lengths.shape[0] == x.shape[0]
        assert x.is_contiguous()

        B, N, D = x.shape
        mask = paddle.arange(max_len).unsqueeze(0).tile([x.shape[0], 1]) < lengths.unsqueeze(1)
        ctx.save_for_backward(mask)
        lengths = lengths.astype(paddle.int32)

        # Previous version (breaks compile graph): 
        # return torch.nested.nested_tensor(
        #    [i[:j.item()] for i, j in zip(x, lengths)],
        #    layout=torch.jagged,
        #    device=x.device,
        #    requires_grad=x.requires_grad
        #)

        offsets = paddle.concat([
            paddle.zeros([1], dtype=lengths.dtype),
            lengths.cumsum(axis=0)
        ])

        jagged_batch_size = lengths.sum().astype(paddle.int32)

        # Initialize empty tensor with right shapes (PaddlePaddle doesn't have nested tensors)
        # Create a simple tensor structure to mimic nested tensor behavior
        target = paddle.empty([jagged_batch_size, D], dtype=x.dtype)
        target_offsets = paddle.empty([len(lengths)+1], dtype=lengths.dtype)
        grid = lambda meta: (B*triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(D, meta['BLOCK_SIZE_D']),)

        _padded_to_jagged_kernel[grid](
            x, lengths, offsets,
            target, target_offsets,
            x.strides[0], x.strides[1], x.strides[2], target.strides[0],
            B, N, D, BLOCK_SIZE_N=32, BLOCK_SIZE_D=D
        )

        # Return a simple tensor instead of nested tensor for PaddlePaddle
        # Store offsets as attribute for compatibility
        target._offsets = target_offsets
        return target
    

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        grad_values = grad_output.values()

        grad_x = paddle.zeros([*mask.shape, grad_values.shape[-1]], dtype=grad_values.dtype)
        grad_x[mask] = grad_values

        return grad_x, None, None


def padded_to_jagged_tensor(x: Tensor, lengths: Tensor, max_len: int) -> NestedTensor:
    """
      Differentiable padded -> Jagged conversion. 
      This will cause a graph break as nested tensor creation is not supported by torch.compile.
    """
    return PaddedToJaggedTensor.apply(x, lengths, max_len)


def jagged_to_flattened_tensor(x: NestedTensor) -> Tensor:
    return x.values()


@triton.jit
def _padded_to_jagged_kernel(
    x_ptr,
    lengths_ptr,
    offsets_ptr,
    out_values_ptr,
    out_offsets_ptr,
    x_stride_B, x_stride_N, x_stride_D,
    out_values_stride_B,
    B, N, D,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    assert BLOCK_SIZE_D == D
    pid_n = tl.program_id(0)
    num_pids_n = tl.cdiv(N, BLOCK_SIZE_N)

    row_group = pid_n // num_pids_n
    col_group = pid_n % num_pids_n

    jagged_row_start_offset, jagged_row_end_offset = tl.load(offsets_ptr + row_group), tl.load(offsets_ptr + row_group + 1)
    padded_tile_start_ptr = x_ptr + row_group*x_stride_B + col_group*BLOCK_SIZE_N*x_stride_N
    padded_tile_offsets = tl.arange(0, end=BLOCK_SIZE_N*BLOCK_SIZE_D)

    max_offset_B = (jagged_row_end_offset - jagged_row_start_offset - col_group*BLOCK_SIZE_N)*x_stride_N
    mask = padded_tile_offsets < max_offset_B
    padded_in_ptr = padded_tile_start_ptr + padded_tile_offsets
    in_values = tl.load(padded_in_ptr, mask=mask)

    out_values_tile_start = out_values_ptr + (jagged_row_start_offset + col_group*BLOCK_SIZE_N)*out_values_stride_B
    out_values_ptr = out_values_tile_start + tl.arange(0, BLOCK_SIZE_N*BLOCK_SIZE_D)
    
    tl.store(out_values_ptr, in_values, mask=mask)
    tl.store(out_offsets_ptr + row_group + tl.arange(0, 2), tl.join(jagged_row_start_offset, jagged_row_end_offset))
