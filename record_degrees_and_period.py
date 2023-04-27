import numpy as np
import math, sys, time, os, glob
from datetime import timedelta
from collections import Counter
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
DATA_DIR = 'record_degrees_and_period'
MAX_DIGITS = 6


def print_progress_bar(start_time, n_trials_done, n_trials_total, my_dict, key1, key2, key3):
  # print progress bar
  sys.stdout.write(
    '\rCompleted: {}/{} trials. {}: {}, {}: {}, {}: {}. Elapsed: {}'.format(
      n_trials_done, n_trials_total,
      status_to_text[key1], my_dict[key1],
      status_to_text[key2], my_dict[key2],
      status_to_text[key3], my_dict[key3],
      timedelta(seconds=time.time() - start_time)))
  sys.stdout.flush()


def trial(n, d, delta):
  g = RandomGraph(n)
  g.set_color(delta)
  g.generate(d/n)
  prev_colors = g.colors
  prev_two_colors = g.colors
  for day in range(1, MAX_DAYS + 1):
    g.transition()

    if np.array_equal(g.colors, prev_colors):
      r_prev, b_prev = g.count(prev_colors)
      r, b = g.count()
      # Number of nodes that are in cycle two at prev day
      r_to_b = 0
      b_to_r = 0
      # The mean has to be -1 then /2 because of the way edges are stored
      avg_r_deg = np.mean(np.sum(g.edges[g.colors == COLOR_RED], axis=1) - 1) / 2
      avg_b_deg = np.mean(np.sum(g.edges[g.colors == COLOR_BLUE], axis=1) - 1) / 2
      avg_r_prev_deg = np.mean(np.sum(g.edges[prev_colors == COLOR_RED], axis=1) - 1) / 2
      avg_b_prev_deg = np.mean(np.sum(g.edges[prev_colors == COLOR_BLUE], axis=1) - 1) / 2
      return (CYCLE_ONE, r_prev, b_prev, r, b, day, r_to_b, b_to_r,
              avg_r_prev_deg, avg_b_prev_deg, avg_r_deg, avg_b_deg)

    if day > 1 and np.array_equal(g.colors, prev_two_colors):
      r_prev, b_prev = g.count(prev_colors)
      r, b = g.count()
      # Number of nodes that are in cycle two at prev day
      r_to_b = np.count_nonzero(np.logical_and(
        g.colors == COLOR_RED, prev_colors == COLOR_BLUE))
      b_to_r = np.count_nonzero(np.logical_and(
        g.colors == COLOR_BLUE, prev_colors == COLOR_RED))
      # The mean has to be -1 then /2 because of the way edges are stored
      avg_r_deg = np.mean(np.sum(g.edges[g.colors == COLOR_RED], axis=1) - 1) / 2
      avg_b_deg = np.mean(np.sum(g.edges[g.colors == COLOR_BLUE], axis=1) - 1) / 2
      avg_r_prev_deg = np.mean(np.sum(g.edges[prev_colors == COLOR_RED], axis=1) - 1) / 2
      avg_b_prev_deg = np.mean(np.sum(g.edges[prev_colors == COLOR_BLUE], axis=1) - 1) / 2
      return (CYCLE_TWO, r_prev, b_prev, r, b, day, r_to_b, b_to_r,
              avg_r_prev_deg, avg_b_prev_deg, avg_r_deg, avg_b_deg)

    prev_two_colors = prev_colors
    prev_colors = g.colors

  r_prev, b_prev = g.count(prev_two_colors)
  r, b = g.count()
  # Number of nodes that are in cycle two at prev day
  r_to_b = 0
  b_to_r = 0
  # The mean has to be -1 then /2 because of the way edges are stored
  avg_r_deg = np.mean(np.sum(g.edges[g.colors == COLOR_RED], axis=1) - 1) / 2
  avg_b_deg = np.mean(np.sum(g.edges[g.colors == COLOR_BLUE], axis=1) - 1) / 2
  avg_r_prev_deg = np.mean(np.sum(g.edges[prev_colors == COLOR_RED], axis=1) - 1) / 2
  avg_b_prev_deg = np.mean(np.sum(g.edges[prev_colors == COLOR_BLUE], axis=1) - 1) / 2
  return (INCONCLUSIVE, r_prev, b_prev, r, b, MAX_DAYS, r_to_b, b_to_r,
          avg_r_prev_deg, avg_b_prev_deg, avg_r_deg, avg_b_deg)


def main(n=10000, d=1, delta=0, n_trials=1000, save=False):
  """
  Runs some trials and print out the number of occurrences for each periodicity
  :param n: number of vertices
  :param d: expected degree (= pn)
  :param delta: initial gap (= |R_0| - |B_0|)
  :param n_trials: number of trials
  :param save: boolean. If set to true, the results will be saved in the designated directory
  DATADIR/[n]_[d]_[delta], where DATADIR is defined at the top
  """
  print('Proceeding to run n_trials = {} trials on random graph G(n, d/n) '
        'for n = {}, d = {}.'.format(n_trials, n, d))
  print('Max days = {}. Advantage c = {}. Starting with {} Reds and {} Blues.'.format(
      MAX_DAYS, delta, math.ceil(n / 2 + delta), math.floor(n / 2 - delta)))
  print()

  # run n_trials trials
  records = []  # dict format: (status, n_red, n_blue): number of occurences
  summary = {CYCLE_ONE: 0, CYCLE_TWO: 0, INCONCLUSIVE: 0}
  start = time.time()
  for i in range(1, n_trials + 1):
    result = trial(n, d, delta)
    records.append(result)
    summary[result[0]] += 1
    print_progress_bar(start, i, n_trials, summary, CYCLE_ONE, CYCLE_TWO, INCONCLUSIVE)
  print()
  print('============== SUMMARY ==================')
  print('Status           Count     Frequency')
  for status, count in summary.items():
    print('{:<12}      {:<6}      {:.2f}'.format(
      status, count, count / n_trials
    ))

  # save records
  if save:
    data_subdir = '{}/{}_{}_{}'.format(DATA_DIR, n, str(d).replace('.', 'd'), delta)
    os.makedirs(data_subdir, exist_ok=True)
    print('Results saved in {}'.format(data_subdir))
    batch_list = [int(fname[- MAX_DIGITS - 4:-4]) for fname in glob.glob(data_subdir + '/*.npy')]
    first_batch = max(batch_list) + 1 if batch_list else 0
    np_records = np.array(records)
    np.save('{}/{:0{}}.npy'.format(data_subdir, first_batch, MAX_DIGITS), records)


# a function to test the main function. Not really needed anymore
def foo(d):
  res = np.load('{}/10000_{}_{}/000000.npy'.format(DATA_DIR, d, 0))
  print('Avg end size: R: {}, B: {}'.format(res[:, 3].mean(), res[:, 4].mean()))
  print('Avg end day: {}'.format(res[:, 5].mean()))
  print('Status           Count            Avg lower size       Avg end day    '
        'Avg R to B    Avg B to R     Avg R degree    Avg B degree')
  for i in range(3):
    indices = res[:, 0] == i
    count = np.count_nonzero(indices)
    res_i = res[indices]
    avg_lower_size = np.mean(np.minimum(res_i[:, 3], res_i[:, 4]))
    avg_end_day = np.mean(res_i[:, 5])
    avg_r_to_b = np.mean(res_i[:, 6])
    avg_b_to_r = np.mean(res_i[:, 7])
    avg_r_deg = np.mean(res_i[:, 10])
    avg_b_deg = np.mean(res_i[:, 11])
    print('{:<12}      {:<6}           {:.2f}             {}       '
          '{:<4}      {:<4}       {:<5}       {:<5}'.format(
      i, count, avg_lower_size, avg_end_day,
      avg_r_to_b, avg_b_to_r, avg_r_deg, avg_b_deg))


if __name__ == '__main__':
  # foo(1)
  main(n=1000, d=10, delta=0, n_trials=10)
