import numpy as np
# import math, sys, time, os, glob
# from datetime import timedelta
# from collections import Counter
import warnings
from scipy.sparse.csgraph import connected_components
from random_graph import RandomGraph

MAX_DAYS = 100
COLOR_RED = 1
COLOR_BLUE = -1
CYCLE_ONE = 1
CYCLE_TWO = 2
INCONCLUSIVE = 0
status_to_text = {
  CYCLE_ONE: 'Cycle one', CYCLE_TWO: 'Cycle_two', INCONCLUSIVE: 'Inconclusive'
}
info_to_index = {
  '|V|': 0, '|E|': 1, 'r(0)': 2, 'b(0)': 3, 'stt': 4,
  'r(t-1)': 5, 'b(t-1)': 6, 'r(t)': 7, 'b(t)': 8,
  '  t': 9, 'r->b': 10, 'b->r': 11, 'r(t-1)_deg': 12,
  'b(t-1)_deg': 13, 'r(t)_deg': 14, 'b(t)_deg': 15
}
DATA_DIR = 'colors_on_components.py'
MAX_DIGITS = 6


# Info to store about each component:
# Size, No of edges, n_red, n_blue, n_red_end, n_blue_end, n_red_prev_end, n_blue_prev_end
# r_to_b, b_to_r, avg_r_deg_end, avg_b_deg_end,
def trial(n, d, c):
  g = RandomGraph(n)
  g.set_color(c)
  g.generate(d/n)
  n_components, component = connected_components(
    np.maximum(g.edges - 1, 0), directed=False, return_labels=True)

  results = np.empty((n_components, 16))
  for k in range(n_components):
    in_g_k = component == k
    size_k = np.count_nonzero(in_g_k)
    g_k = RandomGraph(size_k)
    g_k.edges = g.edges[in_g_k][:, in_g_k]
    g_k.colors = g.colors[in_g_k]
    pre_info = (size_k, np.sum(np.maximum(g_k.edges - 1, 0)) // 2) + g_k.count()
    post_info = trial_on_component(g_k)
    results[k] = pre_info + post_info
  return results


def trial_on_component(g_k):
  prev_colors = g_k.colors
  prev_two_colors = g_k.colors
  status = None
  # Number of nodes that are in cycle two at prev day
  r_to_b = 0
  b_to_r = 0
  day = 0
  for day in range(1, MAX_DAYS + 1):
    prev_colors = g_k.colors
    g_k.transition()
    is_cycle_1 = np.array_equal(g_k.colors, prev_colors)
    is_cycle_2 = day > 1 and np.array_equal(g_k.colors, prev_two_colors)
    if is_cycle_1:
      # Number of nodes that are in cycle two at prev day
      r_to_b = 0
      b_to_r = 0
      status = CYCLE_ONE
      break
    if is_cycle_2:
      # Number of nodes that are in cycle two at prev day
      r_to_b = np.count_nonzero(np.logical_and(
        g_k.colors == COLOR_RED, prev_colors == COLOR_BLUE))
      b_to_r = np.count_nonzero(np.logical_and(
        g_k.colors == COLOR_BLUE, prev_colors == COLOR_RED))
      status = CYCLE_TWO
      break
    prev_two_colors = prev_colors
  if status is None:
    status = INCONCLUSIVE
  # Common calculations
  r_prev, b_prev = g_k.count(prev_colors)
  r, b = g_k.count()
  # The mean has to be -1 then /2 because of the way edges are stored
  avg_r_deg = round(np.mean(np.sum(g_k.edges[g_k.colors == COLOR_RED], axis=1) - 1) / 2, 2)
  avg_b_deg = round(np.mean(np.sum(g_k.edges[g_k.colors == COLOR_BLUE], axis=1) - 1) / 2, 2)
  avg_r_prev_deg = round(np.mean(np.sum(g_k.edges[prev_colors == COLOR_RED], axis=1) - 1) / 2, 2)
  avg_b_prev_deg = round(np.mean(np.sum(g_k.edges[prev_colors == COLOR_BLUE], axis=1) - 1) / 2, 2)
  return (status, r_prev, b_prev, r, b, day, r_to_b, b_to_r,
          avg_r_prev_deg, avg_b_prev_deg, avg_r_deg, avg_b_deg)


def main_single(n=10000, d=1, delta=0, v_threshold=0):
  """
  Run a single trial and print out the information on each connected component
  :param n: total number of vertices
  :param d: expected degree (= pn)
  :param delta: initial gap (= |R_0| - |B_0|)
  :param v_threshold: minimum size required for a component to be displayed in result.
  E.g. if v_threshold = 5, only information on components of size at least 6 will be displayed.
  :return the following infos are printed for each component:
  |V|: number of vertices.
  |E|: number of edges.
  r(0), b(0): number of reds and blues in the beginning in the component.
  stt: periodicity of final state. 1 = stable state. 2 = cycle of 2 states.
  t: first day to enter final state.
  r(t-1), b(t-1): number of reds and blues on day t-1.
  r(t), b(t): number of reds and blues on day t.
  r->b, b->r: number of vertices being red on day t-1 but blue on day t and vice versa.
  r(t-1)_deg, b(t-1)_deg: average degree of reds and blues on day t-1.
  r(t)_deg, b(t)_deg: average degree of reds and blues on day t.
  """
  warnings.filterwarnings('ignore')
  component_infos = trial(n, d, delta)
  component_infos = component_infos[component_infos[:, info_to_index['|V|']] > v_threshold]
  component_infos = component_infos[(-component_infos[:, info_to_index['|V|']]).argsort()]
  # Get proper width for each column
  col_width = {}
  for info in info_to_index.keys():
    col_width[info] = max(len(info), len(str(component_infos[0, info_to_index[info]])))
  # Print headline string
  headline = ''
  for info in info_to_index.keys():
    headline += '{0:>{1}}   '.format(info, col_width[info])
  print(headline)
  # Print each result row
  for k in range(component_infos.shape[0]):
    result = ''
    for info in info_to_index.keys():
      content = component_infos[k, info_to_index[info]]
      if content.is_integer():
        content = int(content)
      # else:
      #   content = round(content, 2)
      result += '{0:>{1}}   '.format(content, col_width[info])
    print(result)


if __name__ == '__main__':
  main_single(n=1000, d=0.9, delta=0, v_threshold=5)

