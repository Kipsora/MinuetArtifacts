__all__ = ['format_table']

from typing import Sequence


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]):
  column_widths = [0] * len(headers)

  rows = list(rows)
  rows.insert(0, headers)
  for i in range(len(rows)):
    rows[i] = list(rows[i])

  num_rows = len(rows)
  num_cols = len(headers)

  for j in range(num_cols):
    for i in range(num_rows):
      column_widths[j] = max(column_widths[j], len(rows[i][j]))

  formatted_rows = []
  for i, row in enumerate(rows):
    formatted_row = "|"
    for j, cell in enumerate(row):
      formatted_row += f' {cell:{column_widths[j]}s} |'
    formatted_rows.append(formatted_row)

    if i == 0:
      formatted_row = '+'
      for j in range(num_cols):
        formatted_row += '=' + ('=' * column_widths[j]) + '=+'
      formatted_rows.append(formatted_row)

  return '\n'.join(formatted_rows)
