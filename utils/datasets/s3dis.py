# Reference: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

__all__ = ['S3DIS']

import os
from typing import Optional

import numpy as np
import torch
import torchsparse.utils.quantize
from torch.utils.data import Dataset


class S3DIS(Dataset):

  def __init__(self,
               split: str = 'test',
               data_root: str = 'data/s3dis',
               num_features: Optional[int] = None,
               dtype: torch.dtype = torch.float32,
               num_point: int = 65536,
               voxel_size: float = 0.005,
               test_area: int = 5,
               block_size: float = 4.0,
               sample_rate: float = 1.0):
    super().__init__()
    self.num_point = num_point
    self.voxel_size = voxel_size
    self.dtype = dtype
    self.block_size = block_size
    self.num_features = num_features
    rooms = sorted(os.listdir(data_root))
    rooms = [room for room in rooms if 'Area_' in room]
    if split == 'train':
      rooms_split = [
          room for room in rooms if not 'Area_{}'.format(test_area) in room
      ]
    else:
      rooms_split = [
          room for room in rooms if 'Area_{}'.format(test_area) in room
      ]

    self.room_points, self.room_labels = [], []
    self.room_coord_min, self.room_coord_max = [], []
    num_point_all = []
    labelweights = np.zeros(13)

    for room_name in rooms_split:
      room_path = os.path.join(data_root, room_name)
      room_data = np.load(room_path)  # xyzrgbl, N*7
      points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
      tmp, _ = np.histogram(labels, range(14))
      labelweights += tmp
      coord_min = np.amin(points, axis=0)[:3]
      coord_max = np.amax(points, axis=0)[:3]
      self.room_points.append(points)
      self.room_labels.append(labels)
      self.room_coord_min.append(coord_min)
      self.room_coord_max.append(coord_max)
      num_point_all.append(labels.size)
    labelweights = labelweights.astype(np.float32)
    labelweights = labelweights / np.sum(labelweights)
    self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
    sample_prob = num_point_all / np.sum(num_point_all)
    num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
    room_idxs = []
    for index in range(len(rooms_split)):
      room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
    self.room_idxs = np.array(room_idxs)

  def __getitem__(self, idx):
    room_idx = self.room_idxs[idx]
    points = self.room_points[room_idx]  # N * 6
    labels = self.room_labels[room_idx]  # N
    N_points = points.shape[0]

    while True:
      center = points[np.random.choice(N_points)][:3]
      block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
      block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
      point_idxs = np.where((points[:, 0] >= block_min[0])
                            & (points[:, 0] <= block_max[0])
                            & (points[:, 1] >= block_min[1])
                            & (points[:, 1] <= block_max[1]))[0]
      if point_idxs.size >= self.num_point:
        break

    if point_idxs.size >= self.num_point:
      selected_point_idxs = np.random.choice(point_idxs,
                                             self.num_point,
                                             replace=False)
    else:
      selected_point_idxs = np.random.choice(point_idxs,
                                             self.num_point,
                                             replace=True)

    # normalize
    selected_points = points[selected_point_idxs, :]  # num_point * 6
    coordinates = np.zeros((self.num_point, 3))
    coordinates[:, 0] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
    coordinates[:, 1] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
    coordinates[:, 2] = selected_points[:, 2] / self.room_coord_max[room_idx][2]

    coordinates -= np.min(coordinates, axis=0, keepdims=True)
    coordinates = torchsparse.utils.quantize.sparse_quantize(
        coordinates, voxel_size=self.voxel_size)
    coordinates = torch.tensor(coordinates)

    result = dict(coordinates=coordinates)
    if self.num_features is not None:
      result['features'] = torch.randn(len(result['coordinates']),
                                       self.num_features,
                                       dtype=self.dtype)
    return result

  def __len__(self):
    return len(self.room_idxs)
