import argparse
import collections
import os.path
import re
from statistics import geometric_mean

from utils.flatplot.bars import BarChart
from matplotlib import pyplot
from matplotlib.ticker import StrMethodFormatter

from minuet.utils.file_system import ensure_directory

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
  return {key: value / 100 for key, value in data.items()}


def parse_csv_line(line: str):
  line = re.findall(r'"(.+?)"', line)
  for i in range(len(line)):
    try:
      line[i] = int(line[i])
      continue
    except ValueError:
      pass
    try:
      line[i] = float(line[i])
      continue
    except ValueError:
      pass

  return line


def load_latency_dict(dataset, batch_size=1, kernel_size=3, *, gpu: str):
  results = dict()
  sizes = []
  for baseline in BASELINES:
    path = f"results/{gpu}/mapping-{dataset}-bs={batch_size}-ks={kernel_size}.{BASELINES[baseline]}.csv"
    if not os.path.exists(path):
      print(
          f"Benchmark results for GPU {gpu}, dataset {dataset}, and baseline {baseline} "
          f"is not found")
      return None, None
    with open(path, "r") as reader:
      lines = reader.readlines()
      last_invalid = 0
      input_size = None
      for i in range(len(lines)):
        if 'input_size' in lines[i]:
          input_size = int(lines[i].split('|')[2])
        if lines[i].startswith("==PROF=="):
          last_invalid = i

      lines = lines[last_invalid + 1:]
      header, rows = lines[0], lines[1:]
      header = parse_csv_line(header)
      rows = list(map(parse_csv_line, rows))
      reports = dict()
      for row in rows:
        report = dict(zip(header, row))
        if 'Metric Value' in report:
          reports[report['Section Name'] + '/'
                  + report['Metric Name']] = report['Metric Value']
      l2_hit_rate = reports.get('Memory Workload Analysis/L2 Hit Rate')
      if l2_hit_rate is None:
        print(f"Benchmark results for GPU {gpu}, dataset {dataset} and "
              f"is not found")
        return None, None
      results[baseline] = l2_hit_rate
      sizes.append(input_size)
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
    axes.yaxis.set_major_formatter(StrMethodFormatter('{x:.0%}'))

    chart = BarChart()

    values = collections.defaultdict(list)
    for label, dataset in DATASETS:
      latency_dict, size = load_latency_dict(dataset, gpu=gpu)
      if latency_dict is None:
        continue
      group_name = f"{size:.2g}"
      base, exp = group_name.split('e+')
      base = float(base)
      exp = int(exp)
      group_name = f"${base}\\times 10^{exp}$\n{label}"
      chart.set_group(group_name, normalize(latency_dict))
      for k, v in latency_dict.items():
        values[k].append(v)

    values = {k: geometric_mean(v) for k, v in values.items()}
    chart.set_group(f"Geomean", normalize(values))

    axes.axvline(x=len(DATASETS) - 1 + 0.5,
                 linestyle="dashed",
                 color="black",
                 linewidth=1)

    def bar_label_format(**kwargs):
      value = kwargs['value']
      return f'{value:.0%}'

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
    axes.set_ylim(ymin=0, ymax=1)
    axes.yaxis.set_major_locator(pyplot.MaxNLocator(nbins=5))
    axes.grid(visible=True, axis="y", linestyle="dashed", zorder=0)

    # chart.intra_group_margin = 0.1
    chart.inter_group_margin = 0.2
    chart.global_extras['zorder'] = 2
    chart.render(axes, stacked=False)

    axes.legend(fontsize=9, ncol=3, loc=(0, 1.1))
    axes.set_ylabel("L2 Cache Hit Ratio")

    figure.tight_layout(pad=0.1)
    ensure_directory(args.output_path)
    filename = f"figure11a_mapping_hit_ratio.{gpu}"
    pyplot.savefig(os.path.join(args.output_path, f"{filename}.pdf"))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_path",
                      type=str,
                      default="./figures",
                      help="output path for the figure")
  main(parser.parse_args())
