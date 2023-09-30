# Reference: https://github.com/mit-han-lab/spvnas/blob/dev/torchsparsepp_backend/core/datasets/semantic_kitti.py

__all__ = ['SemanticKITTI']

import os
import os.path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

label_name_mapping = {
    0: 'unlabeled',
    1: 'outlier',
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    16: 'on-rails',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    40: 'road',
    44: 'parking',
    48: 'sidewalk',
    49: 'other-ground',
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',
    71: 'trunk',
    72: 'terrain',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    256: 'moving-on-rails',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}

kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]


class SemanticKITTI(Dataset):

  def __init__(self,
               path: str = 'data/semantic_kitti/dataset/sequences',
               voxel_size: float = 0.1,
               split: str = 'val',
               num_features: Optional[int] = None,
               dtype: torch.dtype = torch.float32,
               sample_stride=1,
               submit: bool = False,
               google_mode: bool = True):
    if submit:
      trainval = True
    else:
      trainval = False
    self.path = path
    self.split = split
    self.voxel_size = voxel_size
    self.sample_stride = sample_stride
    self.google_mode = google_mode
    self.dtype = dtype
    self.num_features = num_features
    self.seqs = []
    if split == 'train':
      self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
      if self.google_mode or trainval:
        self.seqs.append('08')
    elif self.split == 'val':
      self.seqs = ['08']
    elif self.split == 'test':
      self.seqs = [
          '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
      ]

    self.files = []
    for seq in self.seqs:
      seq_files = sorted(os.listdir(os.path.join(self.path, seq, 'velodyne')))
      seq_files = [
          os.path.join(self.path, seq, 'velodyne', x) for x in seq_files
      ]
      self.files.extend(seq_files)

    if self.sample_stride > 1:
      self.files = self.files[::self.sample_stride]

    reverse_label_name_mapping = {}
    self.label_map = np.zeros(260)
    cnt = 0
    for label_id in label_name_mapping:
      if label_id > 250:
        if label_name_mapping[label_id].replace('moving-', '') in kept_labels:
          self.label_map[label_id] = reverse_label_name_mapping[
              label_name_mapping[label_id].replace('moving-', '')]
        else:
          self.label_map[label_id] = 255
      elif label_id == 0:
        self.label_map[label_id] = 255
      else:
        if label_name_mapping[label_id] in kept_labels:
          self.label_map[label_id] = cnt
          reverse_label_name_mapping[label_name_mapping[label_id]] = cnt
          cnt += 1
        else:
          self.label_map[label_id] = 255

    self.reverse_label_name_mapping = reverse_label_name_mapping
    self.num_classes = cnt
    self.angle = 0.0

  def set_angle(self, angle):
    self.angle = angle

  def __len__(self):
    return len(self.files)

  def __getitem__(self, index):
    with open(self.files[index], 'rb') as b:
      block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
    block = np.zeros_like(block_)

    if 'train' in self.split:
      theta = np.random.uniform(0, 2 * np.pi)
      scale_factor = np.random.uniform(0.95, 1.05)
      rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                          [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

      block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor
    else:
      theta = self.angle
      transform_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])
      block[...] = block_[...]
      block[:, :3] = np.dot(block[:, :3], transform_mat)

    block[:, 3] = block_[:, 3]
    pc_ = np.round(block[:, :3] / self.voxel_size).astype(np.int32)
    pc_ -= pc_.min(0, keepdims=1)

    _, inds, inverse_map = sparse_quantize(pc_,
                                           return_index=True,
                                           return_inverse=True)

    pc = pc_[inds]

    result = dict(coordinates=torch.tensor(pc))
    if self.num_features is not None:
      result['features'] = torch.randn(len(result['coordinates']),
                                       self.num_features,
                                       dtype=self.dtype)
    return result

  @staticmethod
  def collate_fn(inputs):
    return sparse_collate_fn(inputs)
