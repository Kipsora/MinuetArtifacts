import argparse
import collections
import json
import os.path
from statistics import geometric_mean

from matplotlib import pyplot

from minuet.utils.file_system import ensure_directory
from utils.flatplot.bars import BarChart

BASELINES = {
    'MinkowskiEngine': 'minkowski',
    'TorchSparse': 'torchsparse',
    'Minuet': 'minuet',
}
DATASETS = [
    ('Sem3D', 'semantic3d-birdfountain-0.2'),
    ('Sem3D', 'semantic3d-birdfountain-0.1'),
    ('Sem3D', 'semantic3d-birdfountain-0.05'),
    ('Sem3D', 'semantic3d-birdfountain-0.02'),
    ('Random', 'random-1e5-f1-s=0.2'),
    ('Random', 'random-5e5-f1-s=0.2'),
    ('Random', 'random-1e6-f1-s=0.2'),
    ('Random', 'random-5e6-f1-s=0.2'),
]


def normalize(data):
  best = data['MinkowskiEngine']
  return {key: value / best for key, value in data.items()}


def load_latency_dict(dataset, batch_size=1, kernel_size=3, *, gpu: str):
  results = dict()
  sizes = []
  for baseline in BASELINES:
    path = f"results/{gpu}/mapping-{dataset}-bs={batch_size}-ks={kernel_size}.{BASELINES[baseline]}.json"
    if not os.path.exists(path):
      return None, None
    with open(path, "r") as reader:
      data = json.load(reader)
      results[baseline] = data['latency_build']
      sizes.append(data['input_size'])
  assert len(set(sizes)) == 1
  return results, sizes[0]


def main(args):
  GPUS = sorted([
      i for i in os.listdir("results")
      if os.path.isdir(os.path.join(f"results/{i}"))
  ])
  for gpu in GPUS:
    figure: pyplot.Figure = pyplot.figure(figsize=(7, 1.7), dpi=224)
    axes: pyplot.Axes = figure.add_subplot(111)

    chart = BarChart()

    values = collections.defaultdict(list)
    for label, dataset in DATASETS:
      latency_dict, size = load_latency_dict(dataset, gpu=gpu)
      if latency_dict is None:
        print(f"Benchmark results for GPU {gpu}, dataset {dataset} "
              f"is not found")
        continue
      group_name = f"{size:.2g}"
      base, exp = group_name.split('e+')
      base = float(base)
      exp = int(exp)
      group_name = f"${base}\\times 10^{exp}$\n{label}"
      latency_dict = normalize(latency_dict)
      chart.set_group(group_name, latency_dict)
      for k, v in latency_dict.items():
        values[k].append(v)

    values = {k: geometric_mean(v) for k, v in values.items()}
    chart.set_group(f"Geomean", values)

    axes.axvline(x=len(DATASETS) - 1 + 0.5,
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
    axes.yaxis.set_major_locator(pyplot.MaxNLocator(nbins=5))
    axes.set_ylim(ymin=0, ymax=1)
    axes.grid(visible=True, axis="y", linestyle="dashed", zorder=0)

    chart.inter_group_margin = 0.2
    chart.global_extras['zorder'] = 2
    chart.render(axes, stacked=False)

    axes.legend(fontsize=9, ncol=3, loc=(0, 1.1))
    axes.set_ylabel("Normalized Build Time")

    figure.tight_layout(pad=0.1)
    ensure_directory(args.output_path)
    filename = f"figure12_mapping_build_time.{gpu}"
    pyplot.savefig(os.path.join(args.output_path, f"{filename}.pdf"))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_path",
                      type=str,
                      default="./figures",
                      help="output path for the figure")
  main(parser.parse_args())
