def to_time_series(data_by_year):
  """TODO: DOCSTRING"""
  data_series = [[]] * len(data_by_year[0][0])

  for year_elems in data_by_year:
    for pos, elem in enumerate(year_elems):
      data_series[pos] = data_series[pos] + [elem[pos]]

  return data_series