import argparse
import contextlib
import gc
import json
import os
import random
import sys
import tempfile

import minuet
import numpy as np
import torch
from minuet.nn import KernelMapCache
from minuet.utils.file_system import ensure_directory
from scipy.stats import gmean

from utils.datasets.helpers import make_dataset, make_tensor_from_dataset, \
  save_random_state, load_random_state


def main(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.random.manual_seed(args.seed)

  with open(args.dataset, "r") as reader:
    dataset_config = json.load(reader)

  dataset_config['num_features'] = args.channels[0]
  dataset = dataset_config.pop("dataset")
  dataset = make_dataset(dataset, **dataset_config)

  if args.library == "torchsparse":
    from torchsparse.nn import Conv3d
    model = Conv3d(in_channels=args.channels[0],
                   out_channels=args.channels[1],
                   kernel_size=args.kernel_size,
                   stride=1,
                   dilation=1,
                   bias=False)
  elif args.library == "minuet":
    from minuet.nn import SparseConv3d
    model = SparseConv3d(in_channels=args.channels[0],
                         out_channels=args.channels[1],
                         kernel_size=args.kernel_size,
                         stride=1,
                         dilation=1,
                         bias=False)
  elif args.library == "minkowski":
    from MinkowskiEngine import MinkowskiConvolution
    model = MinkowskiConvolution(dimension=3,
                                 in_channels=args.channels[0],
                                 out_channels=args.channels[1],
                                 kernel_size=args.kernel_size,
                                 stride=1,
                                 dilation=1,
                                 bias=False)
  else:
    raise NotImplementedError(args.library)

  model.cuda()
  model.eval()

  timings_full = []
  timings_gmas = []
  os.makedirs(args.cache_path, exist_ok=True)
  dataset_config['batch_size'] = args.batch_size
  model_cache = None
  inputs_factory = make_tensor_from_dataset(dataset,
                                            batch_size=args.batch_size,
                                            library=args.library)
  with torch.no_grad(), tempfile.TemporaryDirectory(
      dir=args.cache_path) as cache_path:
    for r in range(args.num_warmup_rounds + args.num_samples * args.num_rounds):
      if r >= args.num_warmup_rounds and \
              (r - args.num_warmup_rounds) % args.num_rounds == 0:
        inputs_factory = make_tensor_from_dataset(dataset,
                                                  num_features=args.channels[0],
                                                  batch_size=args.batch_size,
                                                  library=args.library)
      inputs = inputs_factory()
      if args.library == "torchsparse" and r == 0:
        import torchsparse
        state = save_random_state()
        torchsparse.backends.benchmark = True
        data = []
        for i in range(args.autotuning_samples):
          new_inputs_factory = make_tensor_from_dataset(
              dataset,
              num_features=args.channels[0],
              batch_size=args.batch_size,
              library=args.library)
          data.append(new_inputs_factory())
        torchsparse.tune(model=model,
                         data_loader=data,
                         n_samples=args.autotuning_samples,
                         enable_fp16=False,
                         save_dir=cache_path)

        minuet.nn.functional.cuda_free_buffers()
        gc.collect()
        torch.cuda.empty_cache()
        load_random_state(state)

      if args.library == "minuet":
        if model_cache is not None:
          model_cache.reset()
        else:
          model_cache = KernelMapCache(ndim=3,
                                       dtype=torch.int32,
                                       device="cuda:0")
          minuet.set_kernel_map_cache(module=model, cache=model_cache)

        tuning_file_path = os.path.join(cache_path, "tuning.json")
        if os.path.exists(tuning_file_path):
          with open(tuning_file_path, "r") as reader:
            minuet.load_tunable_config(module=model, config=json.load(reader))
        else:
          if args.verbose:
            print("Tuning information is not found")
          state = save_random_state()
          data = []
          for i in range(args.autotuning_samples):
            new_inputs_factory = make_tensor_from_dataset(
                dataset, batch_size=args.batch_size, library=args.library)
            tensor = new_inputs_factory()
            from minuet import SparseTensor
            from minuet.nn import functional as F
            index = F.build_sorted_index(tensor.C, batch_dims=tensor.batch_dims)
            tensor = SparseTensor(coordinates=tensor.C[index],
                                  features=tensor.F[index],
                                  batch_dims=tensor.batch_dims)
            data.append(tensor)
          minuet.autotune(model,
                          data=data,
                          model_cache=model_cache,
                          cache_path=tuning_file_path)
          if args.verbose:
            print(f"Tuning information is saved at {tuning_file_path}")
          with open(tuning_file_path, "w") as writer:
            config = minuet.dump_tunable_config(module=model)
            json.dump(config, writer, indent=2)

          minuet.nn.functional.cuda_free_buffers()
          model_cache.reset()
          gc.collect()
          torch.cuda.empty_cache()
          load_random_state(state)

      if r >= args.num_warmup_rounds:
        torch.cuda.synchronize()

        event1 = torch.cuda.Event(enable_timing=True)
        event2 = torch.cuda.Event(enable_timing=True)
        event1.record()
        with torch.cuda.nvtx.range(f"{args.library}.{r}"):
          if args.library == "minuet":
            from minuet.nn import functional as F
            index = F.build_sorted_index(inputs.C, batch_dims=inputs.batch_dims)
            from minuet import SparseTensor
            inputs = SparseTensor(coordinates=inputs.C[index],
                                  features=inputs.F[index],
                                  batch_dims=inputs.batch_dims)
          _ = model(inputs)
        event2.record()
        event2.synchronize()
        timings_full.append(event1.elapsed_time(event2))

        event1 = torch.cuda.Event(enable_timing=True)
        event2 = torch.cuda.Event(enable_timing=True)

        event1.record()
        with torch.cuda.nvtx.range(f"{args.library}.{r}_only"):
          _ = model(inputs)
        event2.record()
        event2.synchronize()
        timings_gmas.append(event1.elapsed_time(event2))
      else:
        with torch.cuda.nvtx.range(f"{args.library}.{r}"):
          if args.library == "minuet":
            from minuet.nn import functional as F
            index = F.build_sorted_index(inputs.C, batch_dims=inputs.batch_dims)
            from minuet import SparseTensor
            inputs = SparseTensor(coordinates=inputs.C[index],
                                  features=inputs.F[index],
                                  batch_dims=inputs.batch_dims)
          _ = model(inputs)

  timings_full = np.asarray(timings_full).reshape(args.num_samples,
                                                  args.num_rounds)
  timings_gmas = np.asarray(timings_gmas).reshape(args.num_samples,
                                                  args.num_rounds)
  result = {
      'dataset': args.dataset,
      'library': args.library,
      'input_size': len(inputs.C),
      'batch_size': args.batch_size,
      'channels': args.channels,
      'kernel_size': args.kernel_size,
      'num_warmup_rounds': args.num_warmup_rounds,
      'num_rounds': args.num_rounds,
      'latency_full': list(gmean(timings_full, axis=-1)),
      'latency_gmas': list(gmean(timings_gmas, axis=-1)),
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
            'latency_full': result['latency_full'],
            'latency_gmas': result['latency_gmas'],
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
  parser.add_argument("--channels",
                      nargs=2,
                      type=int,
                      required=True,
                      help="number of input and output channels")
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
  parser.add_argument("-P",
                      "--num_samples",
                      type=int,
                      default=50,
                      help="number of samples for averaging")
  parser.add_argument("--cache_path",
                      type=str,
                      default=".cache",
                      help="path for caching")
  parser.add_argument("--output",
                      type=str,
                      default=None,
                      help="location for storing results")
  parser.add_argument("--autotuning_samples",
                      default=5,
                      type=int,
                      help="samples for autotuning process")
  parser.add_argument('--verbose',
                      action="store_true",
                      help="be more verbose on the outputs")
  parser.add_argument("--json", action="store_true", help="output json entry")
  main(parser.parse_args())
