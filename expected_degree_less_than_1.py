import numpy as np
import math, sys, time, os, glob
from datetime import timedelta
from collections import Counter
from random_graph import RandomGraph

MAX_DAYS = 100
CYCLE_ONE = 1
CYCLE_TWO = 2
INCONCLUSIVE = 0
status_to_text = {
  CYCLE_ONE: 'Cycle one', CYCLE_TWO: 'Cycle_two', INCONCLUSIVE: 'Inconclusive'
}
DATA_DIR = 'expected_degree_less_than_1'
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


def trial(n, d, c):
  g = RandomGraph(n)
  g.set_color(c)
  g.generate(d/n)
  prev_colors = g.colors
  prev_two_colors = g.colors
  for day in range(1, MAX_DAYS + 1):
    g.transition()
    if np.array_equal(g.colors, prev_colors):
      return (CYCLE_ONE, ) + g.count(prev_colors) + g.count() + (day, )
    if day > 1 and np.array_equal(g.colors, prev_two_colors):
      return (CYCLE_TWO, ) + g.count(prev_colors) + g.count() + (day, )
    prev_two_colors = prev_colors
    prev_colors = g.colors
  return (INCONCLUSIVE, ) + g.count(prev_two_colors) + g.count() + (MAX_DAYS, )


def main(n=10000, d=1, c=0, n_trials=1000):
  print('Proceeding to run n_trials = {} trials on random graph G(n, d/n) '
        'for n = {}, d = {}.'.format(n_trials, n, d))
  print('Max days = {}. Advantage c = {}. Starting with {} Reds and {} Blues.'.format(
      MAX_DAYS, c, math.ceil(n / 2 + c), math.floor(n / 2 - c)))
  print()

  # run n_trials trials
  records = []  # dict format: (status, n_red, n_blue): number of occurences
  summary = {CYCLE_ONE: 0, CYCLE_TWO: 0, INCONCLUSIVE: 0}
  start = time.time()
  for i in range(1, n_trials + 1):
    result = trial(n, d, c)
    records.append(result)
    summary[result[0]] += 1
    print_progress_bar(start, i, n_trials, summary, CYCLE_ONE, CYCLE_TWO, INCONCLUSIVE)
  print()

  # print records
  # print('==================== RESULTS =====================')
  # print('Status           Last day     Count      Frequency')
  # print('---------------------------------------------')
  # records_counter = Counter(records)
  # for result, count in sorted(records_counter.items()):
  #   status, _, _, _, _, day = result
  #   frequency = count / n_trials
  #   # summary[status] += count
  #   print(
  #     '{:<12}      {:<4}         {:<6}      {:.2f}'.format(
  #       status, day, count, frequency
  #     )
  #   )
  print()
  print('============== SUMMARY ==================')
  print('Status           Count     Frequency')
  for status, count in summary.items():
    print('{:<12}      {:<6}      {:.2f}'.format(
      status, count, count / n_trials
    ))

  # save records
  data_subdir = '{}/{}_{}'.format(DATA_DIR, n, str(d).replace('.', 'd'))
  os.makedirs(data_subdir, exist_ok=True)
  print(data_subdir)
  batch_list = [int(fname[- MAX_DIGITS - 4:-4]) for fname in glob.glob(data_subdir + '/*.npy')]
  first_batch = max(batch_list) + 1 if batch_list else 0
  np.save('{}/{:0{}}.npy'.format(data_subdir, first_batch, MAX_DIGITS), records)


def foo(d):
  res = np.load('expected_degree_less_than_1/10000_{}/000000.npy'.format(d))
  print('Avg end size: R: {}, B: {}'.format(res[:, 3].mean(), res[:, 4].mean()))
  print('Avg end day: {}'.format(res[:, 5].mean()))
  print('Status           Count            Avg lower size       Avg end day')
  for i in range(3):
    indices = res[:, 0] == i
    count = np.count_nonzero(indices)
    res_i = res[indices]
    avg_lower_size = np.mean(np.minimum(res_i[:, 3], res_i[:, 4]))
    avg_end_day = np.mean(res_i[:, 5])
    print('{:<12}      {:<6}           {:.2f}               {}'.format(i, count, avg_lower_size, avg_end_day))


if __name__ == '__main__':
  foo(11)
    # for d in range(7, 60):
    #   main(n=10000, d=d, c=0, n_trials=500)
