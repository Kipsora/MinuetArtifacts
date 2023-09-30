# Reference: https://github.com/mit-han-lab/torchsparse/blob/master/torchsparse/backbones/modules/blocks.py

__all__ = ['SparseConvBNReLU', 'SparseResidualBlock']

import MinkowskiEngine as ME
import numpy as np
from minuet.utils.typing import ScalarOrTuple
from torch.nn import Sequential, Module, BatchNorm1d


class MinkowskiBatchNorm(BatchNorm1d):
  r"""A batch normalization layer for a sparse tensor.

  See the pytorch :attr:`torch.nn.BatchNorm1d` for more details.
  """

  def __init__(
      self,
      num_features,
      eps=1e-5,
      momentum=0.1,
      affine=True,
      track_running_stats=True,
  ):
    super(MinkowskiBatchNorm,
          self).__init__(num_features,
                         eps=eps,
                         momentum=momentum,
                         affine=affine,
                         track_running_stats=track_running_stats)

  def forward(self, input):
    output = super().forward(input.F)
    if isinstance(input, ME.TensorField):
      return ME.TensorField(
          output,
          coordinate_field_map_key=input.coordinate_field_map_key,
          coordinate_manager=input.coordinate_manager,
          quantization_mode=input.quantization_mode,
      )
    else:
      return ME.SparseTensor(
          output,
          coordinate_map_key=input.coordinate_map_key,
          coordinate_manager=input.coordinate_manager,
      )

  def __repr__(self):
    s = "({}, eps={}, momentum={}, affine={}, track_running_stats={})".format(
        self.num_features,
        self.eps,
        self.momentum,
        self.affine,
        self.track_running_stats,
    )
    return self.__class__.__name__ + s


class SparseConvBNReLU(Sequential):

  def __init__(self,
               ndim: int,
               in_channels: int,
               out_channels: int,
               kernel_size: ScalarOrTuple[int],
               stride: ScalarOrTuple[int] = 1,
               dilation: ScalarOrTuple[int] = 1,
               bias: bool = False,
               transposed: bool = False,
               eps: float = 1e-5,
               momentum: float = 0.1,
               affine: bool = True,
               track_running_stats: bool = True,
               use_relu: bool = True):
    convolution = ME.MinkowskiConvolutionTranspose if transposed else ME.MinkowskiConvolution
    super().__init__(
        convolution(dimension=ndim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    bias=bias),
        MinkowskiBatchNorm(num_features=out_channels,
                           eps=eps,
                           momentum=momentum,
                           affine=affine,
                           track_running_stats=track_running_stats),
    )
    if use_relu:
      self.append(ME.MinkowskiReLU())


class SparseResidualBlock(Module):

  def __init__(self,
               ndim: int,
               in_channels: int,
               out_channels: int,
               kernel_size: ScalarOrTuple[int],
               stride: ScalarOrTuple[int] = 1,
               dilation: ScalarOrTuple[int] = 1,
               bias: bool = False,
               transposed: bool = False,
               eps: float = 1e-5,
               momentum: float = 0.1,
               affine: bool = True,
               track_running_stats: bool = True):
    super().__init__()
    convolution = ME.MinkowskiConvolutionTranspose if transposed else ME.MinkowskiConvolution
    self.main = Sequential(
        convolution(dimension=ndim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    bias=bias),
        MinkowskiBatchNorm(num_features=out_channels,
                           eps=eps,
                           momentum=momentum,
                           affine=affine,
                           track_running_stats=track_running_stats),
        ME.MinkowskiReLU(),
        convolution(dimension=ndim,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    bias=bias),
    )
    if in_channels != out_channels or np.prod(stride) != 1 or \
            np.prod(dilation) != 1:
      self.shortcut = SparseConvBNReLU(ndim=ndim,
                                       in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=1,
                                       stride=stride,
                                       dilation=dilation,
                                       bias=bias,
                                       transposed=transposed,
                                       eps=eps,
                                       momentum=momentum,
                                       affine=affine,
                                       track_running_stats=track_running_stats,
                                       use_relu=False)
    else:
      self.shortcut = None

    self.relu = ME.MinkowskiReLU()

  def forward(self, x):
    a = self.main(x)
    if self.shortcut is None:
      b = x
    else:
      b = self.shortcut(x)
    x = a + b
    return self.relu(x)
