__all__ = [
    'make_dataset', 'make_tensor_from_dataset', 'make_tensor',
    'load_random_state', 'save_random_state'
]

import functools
import random
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_coordinates_with_batch_dims(
    source: torch.Tensor,
    batch_dims: Optional[torch.Tensor] = None,
    use_last_batch_dim: bool = False):
  target = torch.zeros([source.shape[0], source.shape[1] + 1],
                       device=source.device,
                       dtype=source.dtype)
  if use_last_batch_dim:
    target[:, :-1] = source
    if batch_dims is not None:
      for i in range(len(batch_dims) - 1):
        target[batch_dims[i]:batch_dims[i + 1], -1] = i
  else:
    target[:, 1:] = source
    if batch_dims is not None:
      for i in range(len(batch_dims) - 1):
        target[batch_dims[i]:batch_dims[i + 1], 0] = i
  return target


def make_tensor(coordinates: torch.Tensor,
                features: torch.Tensor,
                batch_dims: Optional[List[int]] = None,
                sort_coordinates_for_minuet: bool = False,
                library: str = "minuet"):
  coordinates = coordinates.contiguous()
  features = features.contiguous()
  if library == "minuet":
    import minuet
    if batch_dims is not None:
      batch_dims = torch.as_tensor(batch_dims,
                                   device=coordinates.device,
                                   dtype=coordinates.dtype)
    if sort_coordinates_for_minuet:
      index = minuet.nn.functional.build_sorted_index(coordinates,
                                                      batch_dims=batch_dims)
      coordinates = coordinates[index]
      features = features[index]
    else:
      features = features.clone()
    return minuet.SparseTensor(coordinates=coordinates,
                               features=features,
                               batch_dims=batch_dims)
  elif library == "minkowski":
    import MinkowskiEngine as ME
    std_coordinates = generate_coordinates_with_batch_dims(
        coordinates, batch_dims, use_last_batch_dim=False)
    return ME.SparseTensor(coordinates=std_coordinates,
                           features=features.clone())
  elif library == "torchsparse":
    import torchsparse as ts
    std_coordinates = generate_coordinates_with_batch_dims(
        coordinates, batch_dims, use_last_batch_dim=True)
    return ts.SparseTensor(coords=std_coordinates, feats=features.clone())
  else:
    raise NotImplementedError(library)


def string_to_dtype(data: str):
  if data == 'fp32':
    return torch.float32
  elif data == 'fp16':
    return torch.float16
  raise NotImplementedError(data)


def make_dataset(dataset, **kwargs):
  if 'dtype' in kwargs:
    kwargs['dtype'] = string_to_dtype(kwargs['dtype'])
  if dataset == "random":
    from utils.datasets.random import RandomDataset
    return RandomDataset(**kwargs)
  elif dataset == "synthetic":
    from utils.datasets.random import SyntheticDataset
    return SyntheticDataset(**kwargs)
  elif dataset == "s3dis":
    from utils.datasets.s3dis import S3DIS
    return S3DIS(**kwargs)
  elif dataset == "semantic3d":
    from utils.datasets.semantic3d import Semantic3D
    return Semantic3D(**kwargs)
  elif dataset == "semantic_kitti":
    from utils.datasets.semantic_kitti import SemanticKITTI
    return SemanticKITTI(**kwargs)
  elif dataset == "BinvoxDataset":
    from utils.datasets.binvox import BinvoxDataset
    return BinvoxDataset(**kwargs)
  else:
    raise NotImplementedError(dataset)


def make_tensor_from_dataset(dataset: Dataset,
                             batch_size: int,
                             num_features: Optional[int] = None,
                             library: Optional[str] = None):
  tensors = []
  indices = np.random.choice(len(dataset), size=batch_size)
  for i in range(batch_size):
    tensors.append(dataset[indices[i]])
  coordinates = [tensor['coordinates'] for tensor in tensors]
  features = []
  for i, tensor in enumerate(tensors):
    if num_features is not None and tensor['features'].shape[1] != num_features:
      features.append(
          torch.rand(len(coordinates[i]), num_features, dtype=torch.float32))
    else:
      features.append(tensor['features'])
  batch_dims = None
  if len(coordinates) > 1:
    batch_dims = [0]
    for x in coordinates:
      batch_dims.append(batch_dims[-1] + len(x))

  coordinates = torch.concat(coordinates, dim=0)
  coordinates = coordinates.cuda()
  features = torch.concat(features, dim=0)
  features = features.cuda()

  kwargs = dict(coordinates=coordinates,
                features=features,
                batch_dims=batch_dims)
  if library is not None:
    kwargs['library'] = library
  return functools.partial(make_tensor, **kwargs)


def save_random_state():
  return {
      'torch': torch.random.get_rng_state(),
      'random': random.getstate(),
      'numpy': np.random.get_state()
  }


def load_random_state(state):
  torch.random.set_rng_state(state['torch'])
  random.setstate(state['random'])
  np.random.set_state(state['numpy'])
