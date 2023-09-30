import os
import pickle
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

SEMANTIC3D_FILENAMES = {
    "birdfountain1": "birdfountain_station1_xyz_intensity_rgb",
    "castleblatten1": "castleblatten_station1_intensity_rgb",
    "castleblatten5": "castleblatten_station5_xyz_intensity_rgb",
    "marketsquarefeldkirch1": "marketplacefeldkirch_station1_intensity_rgb",
    "marketsquarefeldkirch4": "marketplacefeldkirch_station4_intensity_rgb",
    "marketsquarefeldkirch7": "marketplacefeldkirch_station7_intensity_rgb",
    "sg27_3": "sg27_station3_intensity_rgb",
    "sg27_6": "sg27_station6_intensity_rgb",
    "sg27_8": "sg27_station8_intensity_rgb",
    "sg27_10": "sg27_station10_intensity_rgb",
    "sg28_2": "sg28_station2_intensity_rgb",
    "sg28_5": "sg28_station5_xyz_intensity_rgb",
    "stgallencathedral1": "stgallencathedral_station1_intensity_rgb",
    "stgallencathedral3": "stgallencathedral_station3_intensity_rgb",
    "stgallencathedral6": "stgallencathedral_station6_intensity_rgb",
}


def load_semantic3d_points(category: str,
                           path="data/semantic3d",
                           *,
                           voxel_size: float):
  filename = SEMANTIC3D_FILENAMES[category]
  filepath = os.path.join(path, filename + ".txt")

  try:
    import pandas as pd
  except ImportError:
    raise RuntimeError("Analyzing semantic3d points requires the pandas "
                       "package to be installed (Use \"pip install pandas\" "
                       "to install)")

  point_cloud_filepath = os.path.join(path, filename + f"-{voxel_size}.pkl")
  if os.path.exists(point_cloud_filepath):
    with open(point_cloud_filepath, 'rb') as reader:
      points = pickle.load(reader)
  else:
    points = []
    with open(filepath, "r") as reader:
      for line in reader:
        x, y, z, i, r, g, b = line.split()
        points.append((x, y, z))
    from torchsparse.utils.quantize import sparse_quantize
    points = np.asarray(points, dtype=np.float32)
    points = sparse_quantize(points, voxel_size=voxel_size)
    with open(point_cloud_filepath, 'wb') as writer:
      pickle.dump(points, writer)
  return points


class Semantic3D(Dataset):

  def __init__(self,
               category: str,
               fake_size: int = 10000,
               num_features: Optional[int] = None,
               path: str = "data/semantic3d",
               dtype: torch.dtype = torch.float,
               *,
               voxel_size: float):
    self._size = fake_size
    self._path = path
    self._voxel_size = voxel_size
    self._category = category
    self._num_features = num_features
    self._dtype = dtype

  def __len__(self):
    return self._size

  def __getitem__(self, item):
    coordinates = load_semantic3d_points(category=self._category,
                                         voxel_size=self._voxel_size,
                                         path=self._path)
    coordinates = torch.tensor(coordinates, dtype=torch.int32)
    result = dict(coordinates=coordinates)
    if self._num_features is not None:
      result['features'] = torch.randn(len(coordinates),
                                       self._num_features,
                                       dtype=self._dtype)
    return result
