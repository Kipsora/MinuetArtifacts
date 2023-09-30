__all__ = ['BarChart']

from typing import Union, Dict, Optional, Tuple, List, Any, Callable

from matplotlib.axes import Axes

Number = Union[float, int]


class BarSeriesConfig(object):

  def __init__(self, label: str):
    self.label: str = label
    self.extras: Dict[str, Any] = dict()
    self.show_bar_label: bool = False
    self.bar_label_extras: Dict[str, Any] = dict()
    self.bar_label_format: Optional[Callable] = None

  def update(self, **kwargs):
    self.label = kwargs.get('label', self.label)
    self.extras.update(kwargs.get('extras', dict()))
    self.show_bar_label = kwargs.get('show_bar_label', self.show_bar_label)
    self.bar_label_extras.update(kwargs.get('bar_label_extras', dict()))
    self.bar_label_format = kwargs.get("bar_label_format",
                                       self.bar_label_format)


class TickConfig(object):

  def __init__(self, font_dict: Dict[str, Any]):
    self._font_dict = font_dict
    self.rotation: int = 0

  @property
  def font_dict(self):
    return self._font_dict


class AxisConfig(object):

  def __init__(self):
    self._ticks = TickConfig(dict())

  @property
  def ticks(self):
    return self._ticks


class BarGroupConfig(object):

  def __init__(self, label: str):
    self.label: str = label

  def update(self, **kwargs):
    self.label = kwargs.get('label', self.label)


class BarChart(object):

  def __init__(self):
    # data: (series, group) => data
    self._data: Dict[Tuple[str, str], Number] = dict()

    self._series: Dict[str, BarSeriesConfig] = dict()
    self._groups: Dict[str, BarGroupConfig] = dict()

    self._global_extras = dict()

    self.inter_group_margin: float = 0.5
    self.intra_group_margin: float = 0
    self.figure_margin: Union[float, Tuple[float, float]] = 0

    self.xaxis = AxisConfig()

    self.palette = ["#B76C6C", "#7691AD", "#77aca4", "#919191"]

  @property
  def global_extras(self):
    return self._global_extras

  def set_data(self, series: str, group: str, value: Number):
    self._series.setdefault(series, BarSeriesConfig(label=series))
    self._groups.setdefault(group, BarGroupConfig(label=group))
    self._data[(series, group)] = value

  def get_data(self, series: str, group: str):
    return self._data[(series, group)]

  def series(self):
    return self._series.keys()

  def groups(self):
    return self._groups.keys()

  def series_config(self, series: str, **kwargs):
    self._series[series].update(**kwargs)
    return self._series[series]

  def group_config(self, group: str, **kwargs):
    self._groups[group].update(**kwargs)
    return self._groups[group]

  def set_group_label(self,
                      group: str,
                      label: Optional[str] = None,
                      overwrite: bool = True):
    if label is None:
      label = group
    if overwrite:
      self._groups[group] = BarGroupConfig(label)
    else:
      self._groups.setdefault(group, BarGroupConfig(label))

  def set_series_label(self,
                       series: str,
                       label: Optional[str] = None,
                       overwrite: bool = True):
    if label is None:
      label = series
    if overwrite:
      self._series[series] = BarSeriesConfig(label)
    else:
      self._series.setdefault(series, BarSeriesConfig(label))

  def set_series(self, series: str, data: Dict[str, Number]):
    for group, value in data.items():
      self.set_data(series=series, group=group, value=value)

  def set_group(self, group: str, data: Dict[str, Number]):
    for series, value in data.items():
      self.set_data(series=series, group=group, value=value)

  def render(self,
             axes: Axes,
             stacked: bool = False,
             groups_order: Optional[List[str]] = None,
             series_order: Optional[List[str]] = None):
    if groups_order is None:
      groups_order = list(self._groups.keys())
    if series_order is None:
      series_order = list(self._series.keys())

    num_series = len(self._series)
    num_groups = len(self._groups)
    width = 1.0 - self.inter_group_margin
    if not stacked:
      width -= (num_series - 1) * self.intra_group_margin
      width /= num_series

    axes.xaxis.set_ticks(labels=[self._groups[x].label for x in groups_order],
                         ticks=list(range(num_groups)),
                         rotation=self.xaxis.ticks.rotation,
                         fontdict=self.xaxis.ticks.font_dict)

    figure_margin = self.figure_margin
    if isinstance(figure_margin, float):
      figure_margin = (figure_margin, figure_margin)
    if stacked:
      x_min = 0
      x_max = num_groups - 1
      axes.set_xlim(x_min - 0.5 * width - figure_margin[0],
                    x_max + 0.5 * width + figure_margin[1])
    else:
      x_min = 0
      x_max = num_groups - 1
      x_max += (num_series - 1) * (width + self.intra_group_margin)
      x_min -= 0.5 * (num_series * width +
                      (num_series - 1) * self.intra_group_margin)
      x_max -= 0.5 * (num_series * width +
                      (num_series - 1) * self.intra_group_margin)
      axes.set_xlim(x_min - figure_margin[0], x_max + width + figure_margin[1])

    y_reduce_values = dict()
    for series_index, series in enumerate(series_order):
      x_values = []
      y_values = []
      series_config = self._series[series]

      y_bottoms = []
      for group_index, group in enumerate(groups_order):
        value = self._data.get((series, group))
        if value is None:
          continue

        if not stacked:
          x = group_index + series_index * (width + self.intra_group_margin)
          x -= 0.5 * (num_series * width +
                      (num_series - 1) * self.intra_group_margin)
        else:
          x = group_index - 0.5 * width
        x_values.append(x)
        y_values.append(value)

        if stacked:
          y_reduce_values.setdefault(group, 0)
          y_bottoms.append(y_reduce_values[group])
          y_reduce_values[group] += value
        else:
          y_bottoms.append(0)

      extras = self._global_extras.copy()
      extras.update(series_config.extras)
      extras.setdefault('color', self.palette[series_index % len(self.palette)])
      container = axes.bar(x=x_values,
                           height=y_values,
                           width=width,
                           align="edge",
                           bottom=y_bottoms,
                           label=series_config.label,
                           **extras)
      if series_config.show_bar_label:
        labels = None
        if series_config.bar_label_format is not None:
          labels = []
          for group_index, group in enumerate(groups_order):
            value = self._data.get((series, group))
            if value is None:
              continue
            labels.append(
                series_config.bar_label_format(series=series,
                                               group=group,
                                               value=value))
        axes.bar_label(labels=labels,
                       container=container,
                       **series_config.bar_label_extras)
