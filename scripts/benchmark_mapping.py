import argparse
import contextlib
import json
import os
import random
import sys
from typing import Optional, List

import MinkowskiEngine as ME
import numpy as np
import torch
import tqdm
from minuet.utils.file_system import ensure_directory
from minuet.utils.helpers import generate_kernel_offsets
from utils.datasets.helpers import make_dataset


def benchmark_minuet(sources: torch.Tensor,
                     batch_dims: Optional[List[int]] = None,
                     *,
                     kernel_size: int,
                     verbose: bool = False):
  import minuet
  offsets = generate_kernel_offsets(ndim=3,
                                    kernel_size=kernel_size,
                                    source_stride=1,
                                    dilation=1)
  offsets = torch.as_tensor(offsets, dtype=torch.int32, device="cuda:0")
  if batch_dims is not None:
    batch_dims = torch.tensor(batch_dims, dtype=torch.int32, device="cuda:0")

  event1 = torch.cuda.Event(enable_timing=True)
  event2 = torch.cuda.Event(enable_timing=True)
  event3 = torch.cuda.Event(enable_timing=True)
  event4 = torch.cuda.Event(enable_timing=True)

  torch.cuda.current_stream().synchronize()
  event1.record(stream=torch.cuda.current_stream())
  with torch.cuda.nvtx.range("build"):
    index = minuet.nn.functional.build_sorted_index(sources,
                                                    batch_dims=batch_dims)
    sources = sources[index]
  event2.record(stream=torch.cuda.current_stream())
  event2.synchronize()

  sources = sources.contiguous()
  targets = sources.clone()

  torch.cuda.current_stream().synchronize()
  event3.record(stream=torch.cuda.current_stream())
  with torch.cuda.nvtx.range("query"):
    results = minuet.nn.functional.query_sorted_index_with_offsets(
        sources=sources,
        targets=targets,
        offsets=offsets,
        source_batch_dims=batch_dims,
        target_batch_dims=batch_dims)
  event4.record(stream=torch.cuda.current_stream())
  event4.synchronize()

  if verbose:
    print("hit items:", (results != -1).sum().item())
    print("hit rate:",
          (results != -1).sum().item() / len(targets) / len(offsets))

  return {
      'build': event1.elapsed_time(event2),
      'query': event3.elapsed_time(event4)
  }


def benchmark_torchsparse(sources,
                          batch_dims: Optional[torch.Tensor] = None,
                          *,
                          kernel_size: int):
  offsets = generate_kernel_offsets(ndim=3,
                                    kernel_size=kernel_size,
                                    source_stride=1,
                                    dilation=1)
  offsets = torch.tensor(offsets, dtype=sources.dtype, device=sources.device)

  source_padding = torch.zeros(sources.shape[0],
                               1,
                               dtype=sources.dtype,
                               device=sources.device)
  sources = torch.concat([sources, source_padding], dim=-1)
  if batch_dims is not None:
    for i in range(len(batch_dims) - 1):
      sources[batch_dims[i]:batch_dims[i + 1], -1] = i

  event1 = torch.cuda.Event(enable_timing=True)
  event2 = torch.cuda.Event(enable_timing=True)
  event3 = torch.cuda.Event(enable_timing=True)
  event4 = torch.cuda.Event(enable_timing=True)

  from torchsparse.nn import functional as F
  hash_sources = F.sphash(sources.to(torch.int))

  torch.cuda.current_stream().synchronize()
  event1.record(stream=torch.cuda.current_stream())
  hash_table = F.HashTable(hash_sources)
  event2.record(stream=torch.cuda.current_stream())
  event2.synchronize()

  hash_queries = F.sphash(sources, offsets)

  torch.cuda.current_stream().synchronize()
  event3.record(stream=torch.cuda.current_stream())
  _ = hash_table.query(hash_queries)
  event4.record(stream=torch.cuda.current_stream())
  event4.synchronize()

  hash_table.close()

  return {
      'build': event1.elapsed_time(event2),
      'query': event3.elapsed_time(event4)
  }


def benchmark_minkowski(sources: torch.Tensor,
                        batch_dims: Optional[torch.Tensor] = None,
                        *,
                        kernel_size: int):
  source_padding = torch.zeros(sources.shape[0],
                               1,
                               dtype=sources.dtype,
                               device=sources.device)
  sources = torch.concat([source_padding, sources], dim=-1)
  if batch_dims is not None:
    for i in range(len(batch_dims) - 1):
      sources[batch_dims[i]:batch_dims[i + 1], 0] = i

  coordinate_manager = ME.CoordinateManager(
      D=3,
      coordinate_map_type=ME.CoordinateMapType.CUDA,
      allocator_type=None,
      minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
  )

  source_key = ME.CoordinateMapKey([1, 1, 1], "source")
  target_key = ME.CoordinateMapKey([1, 1, 1], "target")

  target_key, _ = coordinate_manager.insert_and_map(sources,
                                                    *target_key.get_key())

  event1 = torch.cuda.Event(enable_timing=True)
  event2 = torch.cuda.Event(enable_timing=True)
  event3 = torch.cuda.Event(enable_timing=True)
  event4 = torch.cuda.Event(enable_timing=True)

  event1.record(stream=torch.cuda.current_stream())
  source_key, _ = coordinate_manager.insert_and_map(sources,
                                                    *source_key.get_key())
  event2.record(stream=torch.cuda.current_stream())
  event2.synchronize()

  event3.record(stream=torch.cuda.current_stream())
  _ = coordinate_manager.kernel_map(in_key=source_key,
                                    out_key=target_key,
                                    kernel_size=kernel_size)
  event4.record(stream=torch.cuda.current_stream())
  event4.synchronize()

  return {
      'build': event1.elapsed_time(event2),
      'query': event3.elapsed_time(event4)
  }


def make_new_data(dataset, batch_size):
  tensors = []
  indices = np.random.choice(len(dataset), size=batch_size)
  for i in range(batch_size):
    tensors.append(dataset[indices[i]])
  coordinates = [tensor['coordinates'] for tensor in tensors]

  batch_dims = None
  if len(coordinates) > 1:
    batch_dims = [0]
    for x in coordinates:
      batch_dims.append(batch_dims[-1] + len(x))

  coordinates = torch.concat(coordinates, dim=0)
  coordinates = coordinates.cuda()
  return coordinates, batch_dims


def main(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.random.manual_seed(args.seed)

  with open(args.dataset, "r") as reader:
    dataset_config = json.load(reader)

  dataset = dataset_config.pop("dataset")
  dataset_config['num_features'] = None
  dataset = make_dataset(dataset, **dataset_config)
  dataset_config['batch_size'] = args.batch_size

  build_timings = []
  query_timings = []
  coordinates, batch_dims = make_new_data(dataset, batch_size=args.batch_size)
  for r in tqdm.trange(args.num_warmup_rounds + args.num_rounds,
                       leave=False,
                       dynamic_ncols=True):
    if args.library == "minuet":
      result = benchmark_minuet(sources=coordinates,
                                batch_dims=batch_dims,
                                kernel_size=args.kernel_size)
    elif args.library == "torchsparse":
      try:
        result = benchmark_torchsparse(sources=coordinates,
                                       batch_dims=batch_dims,
                                       kernel_size=args.kernel_size)
      except torch.cuda.OutOfMemoryError:
        print("OOM Error occurred")
        return
    elif args.library == "minkowski":
      try:
        result = benchmark_minkowski(sources=coordinates,
                                     batch_dims=batch_dims,
                                     kernel_size=args.kernel_size)
      except MemoryError:
        print("OOM Error occurred")
        return
    else:
      raise NotImplementedError(args.library)

    if r >= args.num_warmup_rounds:
      build_timings.append(result['build'])
      query_timings.append(result['query'])

  result = {
      'dataset': args.dataset,
      'library': args.library,
      'input_size': len(coordinates),
      'batch_size': args.batch_size,
      'kernel_size': args.kernel_size,
      'num_warmup_rounds': args.num_warmup_rounds,
      'num_rounds': args.num_rounds,
      'latency_build': np.average(build_timings),
      'latency_query': np.average(query_timings),
      'dataset_config': dataset_config
  }

  if args.output is not None:
    ensure_directory(os.path.dirname(args.output))
  with contextlib.nullcontext(enter_result=sys.stdout) \
          if args.output is None else open(args.output, "w") as writer:
    if args.json:
      writer.write(json.dumps(result, indent=2) + '\n')
    else:
      from utils.helpers import format_table
      result.pop('dataset_config')
      if not args.verbose:
        result = {
            'latency_build': result['latency_build'],
            'latency_query': result['latency_query'],
        }
      rows = list(map(lambda e: tuple(map(str, e)), result.items()))
      writer.write(format_table(headers=['Field', 'Value'], rows=rows) + '\n')


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
                      "--library",
                      type=str,
                      required=True,
                      help="the library to be benchmarked")
  parser.add_argument("-K",
                      "--kernel_size",
                      type=int,
                      required=True,
                      help="kernel size of the convolution layer")
  parser.add_argument("-D",
                      "--dataset",
                      type=str,
                      required=True,
                      help="dataset for benchmarking")
  parser.add_argument("-W",
                      "--num_warmup_rounds",
                      type=int,
                      default=3,
                      help="number of rounds for warmup")
  parser.add_argument("-T",
                      "--num_rounds",
                      type=int,
                      default=10,
                      help="number of rounds for averaging")
  parser.add_argument("--cache_path",
                      type=str,
                      default=".cache",
                      help="path for caching")
  parser.add_argument("--output",
                      type=str,
                      default=None,
                      help="location for storing results")
  parser.add_argument('--verbose',
                      action="store_true",
                      help="be more verbose on the outputs")
  parser.add_argument("--json", action="store_true", help="output json entry")
  main(parser.parse_args())
