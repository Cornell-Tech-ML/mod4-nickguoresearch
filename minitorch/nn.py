from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw
    output = input.contiguous().view(batch, channel, height, new_width, kw)
    output = output.permute(0, 1, 3, 2, 4)
    output = output.contiguous().view(batch, channel, new_width, new_height, kh * kw)
    output = output.permute(0, 1, 3, 2, 4)
    return output, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D"""
    batch, channel, height, width = input.shape
    input, new_height, new_width = tile(input, kernel)
    pooled = input.mean(dim=-1)
    return pooled.view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a tensor"""
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward of max"""
        d = int(dim[0])
        ctx.save_for_backward(input, d)
        return max_reduce(input, d)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max"""
        (input, dim) = ctx.saved_values
        return argmax(input, dim) * grad_output, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Max function"""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor"""
    in_exp = input.exp()
    return in_exp / in_exp.sum(dim=dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor"""
    m = max(input, dim=dim)
    return input - m - (input - m).exp().sum(dim=dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D"""
    batch, channel, height, width = input.shape
    input, new_height, new_width = tile(input, kernel)
    out = max(input, dim=4).view(batch, channel, new_height, new_width)
    return out


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout based on random"""
    if not ignore:
        return input * (rand(input.shape) > rate)
    return input
