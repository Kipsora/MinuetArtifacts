__all__ = ['RandomDataset', 'SyntheticDataset']

from typing import Optional

import numpy as np

import torch
from torch.utils.data import Dataset
from minuet.utils.random import random_points


class RandomDataset(Dataset):

  def __init__(self,
               min_points: int,
               max_points: int,
               size: int,
               num_features: Optional[int] = None,
               sparsity: float = 0.2,
               ndim: int = 3,
               dtype: torch.dtype = torch.float):
    self._size = size
    self._ndim = ndim
    self._sparsity = sparsity
    self._min_points = min_points
    self._max_points = max_points
    self._num_features = num_features
    self._dtype = dtype

  def __len__(self):
    return self._size

  def __getitem__(self, item):
    num_points = np.random.randint(self._min_points, self._max_points + 1)
    c_max = np.ceil(np.power(num_points / self._sparsity, 1. / self._ndim))
    coordinates = random_points(ndim=self._ndim,
                                n=num_points,
                                c_min=0,
                                c_max=c_max)
    coordinates = torch.tensor(coordinates, dtype=torch.int32)
    result = dict(coordinates=coordinates)
    if self._num_features is not None:
      result['features'] = torch.randn(len(result['coordinates']),
                                       self._num_features,
                                       dtype=self._dtype)
    return result


class SyntheticDataset(Dataset):

  def __init__(self,
               num_points: int,
               c_max: int,
               ndim: int = 3,
               num_features: Optional[int] = None,
               dtype: torch.dtype = torch.float):
    self._size = num_points
    self._c_max = c_max
    self._ndim = ndim
    self._num_features = num_features
    self._dtype = dtype

  def __len__(self):
    return self._size

  def __getitem__(self, item):
    coordinates = random_points(ndim=self._ndim,
                                n=self._size,
                                c_min=0,
                                c_max=self._c_max)
    coordinates = torch.tensor(coordinates, dtype=torch.int32)
    result = dict(coordinates=coordinates)
    if self._num_features is not None:
      result['features'] = torch.randn(len(result['coordinates']),
                                       self._num_features,
                                       dtype=self._dtype)
    return result
