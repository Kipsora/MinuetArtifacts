# Reference: https://github.com/dimatura/binvox-rw-py/blob/public/binvox_rw.py

__all__ = ['BinvoxDataset']

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class BinvoxDataset(Dataset):

  def __init__(self,
               root: str,
               num_features: Optional[int] = None,
               num_points: Optional[int] = None,
               dtype: torch.dtype = torch.float32):
    self._paths = []
    self.num_features = num_features
    self.dtype = dtype
    for filename in os.listdir(root):
      if os.path.splitext(filename)[1] == ".binvox":
        self._paths.append(os.path.join(root, filename))
    self._paths = sorted(self._paths)
    self._num_points = num_points

  def __len__(self):
    return len(self._paths)

  @classmethod
  def _read_header(cls, reader):
    line = reader.readline().strip()
    if not line.startswith(b'#binvox'):
      raise IOError('Not a binvox file')
    dims = [int(i) for i in reader.readline().strip().split(b' ')[1:]]
    translate = [float(i) for i in reader.readline().strip().split(b' ')[1:]]
    scale = [float(i) for i in reader.readline().strip().split(b' ')[1:]][0]
    reader.readline()
    return dims, translate, scale

  def __getitem__(self, item):
    with open(self._paths[item], 'rb') as reader:
      dims, translate, scale = self._read_header(reader)
      raw_data = np.frombuffer(reader.read(), dtype=np.uint8)
      values, counts = raw_data[::2], raw_data[1::2]

    end_indices = np.cumsum(counts)
    indices = np.concatenate(([0], end_indices[:-1])).astype(end_indices.dtype)

    values = values.astype(bool)
    indices = indices[values]
    end_indices = end_indices[values]

    nz_voxels = []
    for index, end_index in zip(indices, end_indices):
      nz_voxels.extend(range(index, end_index))
    nz_voxels = np.array(nz_voxels)

    x = nz_voxels // (dims[0] * dims[1])
    zwpy = nz_voxels % (dims[0] * dims[1])
    z = zwpy // dims[0]
    y = zwpy % dims[0]

    coordinates = np.stack((x, y, z), axis=1)
    from torchsparse.utils.quantize import sparse_quantize
    coordinates = sparse_quantize(coordinates)

    if self._num_points is not None and len(coordinates) > self._num_points:
      indices = np.random.choice(np.arange(len(coordinates)),
                                 self._num_points,
                                 replace=False)
      coordinates = coordinates[indices]

    coordinates = torch.tensor(coordinates, dtype=torch.int32)
    result = dict(coordinates=coordinates)
    if self.num_features is not None:
      result['features'] = torch.randn(len(result['coordinates']),
                                       self.num_features,
                                       dtype=self.dtype)
    return result
