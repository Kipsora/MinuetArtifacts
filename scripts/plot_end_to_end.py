import argparse
import collections
import json
import os.path
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
    'Shape': 'shapenetsem',
}
DATASET_PATHS = {
    'KITTI': 'semantic_kitti',
    'S3DIS': 's3dis',
    'Sem3D': 'semantic3d',
    'Shape': 'shapenet',
}
MODELS = {
    'ResNet': 'SparseResNet21D',
    'UNet': 'SparseResUNet42',
}


def normalize(data):
  best = data['MinkowskiEngine']
  return {key: best / value for key, value in data.items()}


def load_latency_dict(dataset, model, gpu):
  results = dict()
  for baseline in BASELINES:
    path = f"results/{gpu}/{DATASETS[dataset]}-bs=1-{MODELS[model]}.{BASELINES[baseline]}.json"
    with open(path, "r") as reader:
      results[baseline] = np.mean(json.load(reader)['latency'])
  return results


def main(args):
  avg_speedup_over_gpus = collections.defaultdict(list)
  GPUS = sorted([
      i for i in os.listdir("results")
      if os.path.isdir(os.path.join(f"results/{i}"))
  ])
  for gpu in GPUS:
    figure: pyplot.Figure = pyplot.figure(figsize=(5, 1.7), dpi=224)
    axes: pyplot.Axes = figure.add_subplot(111)

    chart = BarChart()

    chart.palette = list(reversed(chart.palette))
    chart.palette = [chart.palette[0], chart.palette[1], chart.palette[3]]

    values = collections.defaultdict(list)
    for i, model in enumerate(MODELS):
      for dataset in DATASETS:
        if os.path.isdir(os.path.join("data", DATASET_PATHS[dataset])):
          latency_dict = load_latency_dict(dataset, model, gpu)
          chart.set_group(f"{dataset}\n{model}", normalize(latency_dict))
          for k, v in latency_dict.items():
            values[k].append(v)

    # Print Macros
    geomean = dict()

    avg_speedup_over_baselines = []
    for baseline in BASELINES:
      if baseline != "Minuet":
        speedup = [a / b for a, b in zip(values[baseline], values['Minuet'])]
        speedup_avg = geometric_mean(speedup)
        avg_speedup_over_baselines.append(speedup_avg)
        avg_speedup_over_gpus[baseline].extend(speedup)
      geomean[baseline] = [
          a / b for a, b in zip(values['MinkowskiEngine'], values[baseline])
      ]

    geomean = {k: geometric_mean(v) for k, v in geomean.items()}
    chart.set_group(f"Geomean", geomean)

    axes.axvline(x=len(MODELS) * len(DATASETS) - 1 + 0.5,
                 linestyle="dashed",
                 color="black",
                 linewidth=1)

    def bar_label_format(**kwargs):
      value = kwargs['value']
      return f'{value:.2f}'

    chart.series_config("TorchSparse",
                        show_bar_label=True,
                        bar_label_format=bar_label_format,
                        bar_label_extras={'fontsize': 8})
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
    axes.yaxis.set_major_locator(pyplot.MaxNLocator(nbins=5))
    axes.grid(visible=True, axis="y", linestyle="dashed", zorder=0)

    # chart.intra_group_margin = 0.1
    chart.inter_group_margin = 0.2
    chart.global_extras['zorder'] = 2
    chart.render(axes, stacked=False)

    axes.legend(fontsize=9, ncol=3, loc=(0, 1.1))
    axes.set_ylabel("Speedup")

    figure.tight_layout(pad=0.1)

    ensure_directory(args.output_path)
    filename = f"figure9_end_to_end_speed_up.{gpu}"
    pyplot.savefig(os.path.join(args.output_path, f"{filename}.pdf"))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_path",
                      type=str,
                      default="./figures",
                      help="output path for the figure")
  main(parser.parse_args())
