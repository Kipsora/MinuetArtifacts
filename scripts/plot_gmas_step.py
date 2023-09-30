import argparse
import collections
import json
import os
from statistics import geometric_mean

import numpy as np
from matplotlib import pyplot

from minuet.utils.file_system import ensure_directory
from utils.flatplot.bars import BarChart

BASELINES = {
    'MinkowskiEngine': 'minkowski',
    'TorchSparse': 'torchsparse',
    'Minuet': 'minuet',
}
DATASETS = {
    'KITTI': 'semantic_kitti',
    'S3DIS': 's3dis',
    'Sem3D': 'semantic3d-birdfountain-0.2',
    'Shape': 'shapenetsem'
}
DATASET_PATHS = {
    'KITTI': 'semantic_kitti',
    'S3DIS': 's3dis',
    'Sem3D': 'semantic3d',
    'Shape': 'shapenet'
}

CHANNELS = {
    '(4,16)': '4x16',
    '(8,8)': '8x8',
    '(16,16)': '16x16',
    '(32,32)': '32x32',
    '(64,64)': '64x64',
    '(64,96)': '64x96',
    '(128,128)': '128x128',
    '(256,256)': '256x256',
    '(256,384)': '256x384',
}


def normalize(data):
  best = data['MinkowskiEngine']
  return {key: best / value for key, value in data.items()}


def load_latency_dict(dataset, channels, kernel_size, *, gpu: str):
  results = dict()
  for baseline in BASELINES:
    path = f"results/{gpu}/{DATASETS[dataset]}-bs=1-c={CHANNELS[channels]}-ks={kernel_size}.{BASELINES[baseline]}.json"
    with open(path, "r") as reader:
      results[baseline] = np.mean(json.load(reader)['latency_gmas'])
  return results


def plot_gpu(args, gpu: str):
  figure: pyplot.Figure = pyplot.figure(figsize=(5, 1.7), dpi=224)
  axes: pyplot.Axes = figure.add_subplot(111)

  chart = BarChart()

  values = collections.defaultdict(list)
  for i, channels in enumerate(CHANNELS):
    group_name = f"{channels}"
    channel_values = collections.defaultdict(list)
    for dataset in DATASETS:
      if not os.path.isdir(os.path.join("data", DATASET_PATHS[dataset])):
        continue
      try:
        latency_dict = load_latency_dict(dataset,
                                         channels=channels,
                                         kernel_size=3,
                                         gpu=gpu)
      except FileNotFoundError:
        print(f"Benchmark results for GPU {gpu}, dataset {dataset} and "
              f"channels {channels} is not found")
        return
      for k, v in latency_dict.items():
        channel_values[k].append(v)

    channel_values = {k: geometric_mean(v) for k, v in channel_values.items()}

    chart.set_group(group_name, normalize(channel_values))
    for k, v in channel_values.items():
      values[k].append(v)

  geomean = dict()

  avg_speedup_over_baselines = []
  for baseline in BASELINES:
    if baseline != "Minuet":
      speedup = [a / b for a, b in zip(values[baseline], values['Minuet'])]
      speedup_avg = geometric_mean(speedup)
      avg_speedup_over_baselines.append(speedup_avg)
    geomean[baseline] = [
        a / b for a, b in zip(values['MinkowskiEngine'], values[baseline])
    ]

  geomean = {k: geometric_mean(v) for k, v in geomean.items()}
  chart.set_group(f"Geomean", geomean)

  axes.axvline(x=len(CHANNELS) - 1 + 0.5,
               linestyle="dashed",
               color="black",
               linewidth=1)

  def bar_label_format(**kwargs):
    value = kwargs['value']
    if value < 1:
      return f'{value:.2g}'
    else:
      return f'{value:.3g}'

  chart.palette = list(reversed(chart.palette))
  if len(chart.series()) == 2:
    chart.palette = [chart.palette[1], chart.palette[3]]
  else:
    chart.palette = [chart.palette[0], chart.palette[1], chart.palette[3]]

  chart.series_config("TorchSparse",
                      show_bar_label=True,
                      bar_label_format=bar_label_format,
                      bar_label_extras={'fontsize': 8})
  if "MinkowskiEngine" in chart.series():
    chart.series_config("MinkowskiEngine",
                        show_bar_label=True,
                        bar_label_format=bar_label_format,
                        bar_label_extras={'fontsize': 8})
  chart.series_config("Minuet",
                      show_bar_label=True,
                      bar_label_format=bar_label_format,
                      bar_label_extras={'fontsize': 8})

  chart.figure_margin = 0.2, 0.2

  chart.xaxis.ticks.font_dict.update({'size': 9})
  chart.xaxis.ticks.rotation = 15
  axes.yaxis.set_major_locator(pyplot.MaxNLocator(nbins=5))
  axes.grid(visible=True, axis="y", linestyle="dashed", zorder=0)
  axes.set_xlabel("Layer $(C_\\mathrm{in}, C_\\mathrm{out})$",
                  fontdict={'size': 11})

  chart.inter_group_margin = 0.2
  chart.global_extras['zorder'] = 2
  chart.render(axes, stacked=False)

  axes.legend(fontsize=9, ncol=3, loc=(0, 1.1))
  axes.set_ylabel("Speedup")

  figure.tight_layout(pad=0.1)
  ensure_directory(args.output_path)
  filename = f"figure14_gather_gemm_scatter_layerwise_speedup.{gpu}"
  pyplot.savefig(os.path.join(args.output_path, f"{filename}.pdf"))


def main(args):
  GPUS = sorted([
      i for i in os.listdir("results")
      if os.path.isdir(os.path.join(f"results/{i}"))
  ])
  for gpu in GPUS:
    plot_gpu(args, gpu)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_path",
                      type=str,
                      default="./figures",
                      help="output path for the figure")
  main(parser.parse_args())
