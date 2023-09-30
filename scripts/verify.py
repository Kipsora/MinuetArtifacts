import argparse
import json
import random

import numpy as np
import torch

import minuet
from minuet.nn import KernelMapCache
from utils.datasets.helpers import make_tensor_from_dataset, make_dataset, make_tensor


def main(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.random.manual_seed(args.seed)

  with open(args.dataset, "r") as reader:
    dataset_config = json.load(reader)

  dataset = dataset_config.pop("dataset")
  dataset_config['num_features'] = 4
  dataset = make_dataset(dataset, **dataset_config)
  dataset_config['batch_size'] = args.batch_size

  if args.model == "SparseResUNet42":
    from utils.models.minkowski import SparseResUNet42 as MinkowskiModel
    from utils.models.minuet import SparseResUNet42 as SRCModel
    from utils.models.torchsparse import SparseResUNet42 as TorchSparseModel
  elif args.model == "SparseResNet21D":
    from utils.models.minkowski import SparseResNet21D as MinkowskiModel
    from utils.models.minuet import SparseResNet21D as SRCModel
    from utils.models.torchsparse import SparseResNet21D as TorchSparseModel
  else:
    raise NotImplementedError(args.model)

  src_model = SRCModel()
  src_model.cuda()
  src_model.eval()
  src_cache = KernelMapCache(ndim=3,
                             dtype=torch.int32,
                             device="cuda:0",
                             layout=args.baseline)
  minuet.set_kernel_map_cache(src_model, src_cache)

  if args.baseline == "minkowski":
    std_model = MinkowskiModel()
  elif args.baseline == "torchsparse":
    std_model = TorchSparseModel()
  else:
    raise NotImplementedError(args.baseline)

  std_model.cuda()
  std_model.eval()

  for i, (key, value) in enumerate(src_model.named_parameters()):
    dict(std_model.named_parameters())[key].data[:] = value.data[:]

  with torch.no_grad():
    for i in range(args.num_tests):
      input_factory = make_tensor_from_dataset(dataset,
                                               batch_size=args.batch_size)

      # SRC
      src_inputs = input_factory(library="minuet")
      src_cache.reset()

      torch.cuda.synchronize()
      with torch.cuda.nvtx.range("minuet"):
        from minuet.nn import functional as F

        index = F.build_sorted_index(src_inputs.C,
                                     batch_dims=src_inputs.batch_dims)
        src_inputs = make_tensor(coordinates=src_inputs.C[index],
                                 features=src_inputs.F[index],
                                 batch_dims=src_inputs.batch_dims,
                                 library="minuet")
        if i == 0:
          minuet.autotune(src_model, data=[src_inputs], model_cache=src_cache)
        src_outputs = src_model(src_inputs)[-1]

      torch.cuda.synchronize()
      src_coordinates = src_outputs.C
      src_features = src_outputs.F
      index = minuet.nn.functional.build_sorted_index(
          src_coordinates, batch_dims=src_outputs.batch_dims)
      src_coordinates = src_coordinates[index]
      src_features = src_features[index]
      torch.cuda.synchronize()

      # STD
      std_inputs = input_factory(library=args.baseline)
      torch.cuda.synchronize()
      with torch.cuda.nvtx.range(args.baseline):
        std_outputs = std_model(std_inputs)[-1]

      torch.cuda.synchronize()
      if args.baseline == "minkowski":
        std_coordinates = std_outputs.C[:, 1:]
      else:
        std_coordinates = std_outputs.C[:, :-1]
      std_features = std_outputs.F

      index = minuet.nn.functional.build_sorted_index(
          std_coordinates, batch_dims=src_outputs.batch_dims)
      std_coordinates = std_coordinates[index]
      std_features = std_features[index]
      torch.cuda.synchronize()

      print(f"Test {i}:")
      print(
          f"Coordinates are equal? {(src_coordinates == std_coordinates).all().item()}"
      )
      print(
          f"Features have less than {args.eps} error? "
          f"{torch.less(torch.abs(src_features - std_features), args.eps).all().item()}"
      )
      print()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-S",
                      "--seed",
                      type=int,
                      default=0,
                      help="seed for randomness control")
  parser.add_argument("-B",
                      "--batch_size",
                      type=int,
                      default=1,
                      help="batch size (> 1 for batch inference)")
  parser.add_argument("-L",
                      "--baseline",
                      type=str,
                      required=True,
                      choices=['minkowski', 'torchsparse'],
                      help="the baseline library to be compared to")
  parser.add_argument("-D",
                      "--dataset",
                      type=str,
                      required=True,
                      help="dataset for benchmarking")
  parser.add_argument("-T",
                      "--num_tests",
                      type=int,
                      default=5,
                      help="number of test")
  parser.add_argument(
      "--eps",
      default=1e-6,
      type=float,
      help="max error for verifying the equality of two floating numbers")
  parser.add_argument("-M",
                      "--model",
                      type=str,
                      help="the model to be verified")
  main(parser.parse_args())
