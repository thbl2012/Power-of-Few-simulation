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
DATA_DIR = 'optimal_power_of_few'
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


def trial(n, p, c):
  g = RandomGraph(n)
  g.set_color(c)
  g.generate(p)
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


def main(n=10000, f=1, c=1, n_trials=1000):
  p = f/c**2
  print('Proceeding to run n_trials = {} trials on random graph G(n, p) '
        'for n = {}, p = {}/c^2 = {}.'.format(n_trials, n, f, p))
  print('Max days = {}. Advantage c = {}. Starting with {} Reds and {} Blues.'.format(
      MAX_DAYS, c, math.ceil(n / 2 + c), math.floor(n / 2 - c)))
  print()

  # run n_trials trials
  records = []  # dict format: (status, n_red, n_blue): number of occurences
  summary = {CYCLE_ONE: 0, CYCLE_TWO: 0, INCONCLUSIVE: 0}
  start = time.time()
  for i in range(1, n_trials + 1):
    result = trial(n, p, c)
    records.append(result)
    summary[result[0]] += 1
    print_progress_bar(start, i, n_trials, summary, CYCLE_ONE, CYCLE_TWO, INCONCLUSIVE)
  print()
  print()
  print('============== SUMMARY ==================')
  print('Status           Count     Frequency')
  for status, count in summary.items():
    print('{:<12}      {:<6}      {:.2f}'.format(
      status, count, count / n_trials
    ))

  # save records
  data_subdir = '{}/{}_{}_{}'.format(DATA_DIR, n, f, c)
  os.makedirs(data_subdir, exist_ok=True)
  print(data_subdir)
  batch_list = [int(fname[- MAX_DIGITS - 4:-4]) for fname in glob.glob(data_subdir + '/*.npy')]
  first_batch = max(batch_list) + 1 if batch_list else 0
  np.save('{}/{:0{}}.npy'.format(data_subdir, first_batch, MAX_DIGITS), records)


def foo(f, c):
  res = np.load('{}/10000_{}_{}/000000.npy'.format(DATA_DIR, f, c))
  # print('Avg end size: R: {}, B: {}'.format(res[:, 3].mean(), res[:, 4].mean()))
  # print('Avg end day: {}'.format(res[:, 5].mean()))
  print('f = {}, c = {}'.format(f, c))
  print('Status     Count      Freq      Avg lower    Avg end')
  for i in range(3):
    indices = res[:, 0] == i
    res_i = res[indices]
    if i == CYCLE_ONE:
      ind_w = np.minimum(res_i[:, 3], res_i[:, 4]) == 0
      res_1w = res_i[ind_w]
      res_1s = res_i[np.logical_not(ind_w)]
      for l, result in zip(['1w', '1s'], [res_1w, res_1s]):
        count = result.shape[0]
        avg_lower_size = np.mean(np.minimum(result[:, 3], result[:, 4]))
        avg_end_day = np.mean(result[:, 5])
        print('{:<9}   {:<6}    {:<6.2f}      {:<8.2f}    {:<8.2f}'.format(
          l, count, count / res.shape[0], avg_lower_size, avg_end_day))
    else:
      count = np.count_nonzero(indices)
      avg_lower_size = np.mean(np.minimum(res_i[:, 3], res_i[:, 4]))
      avg_end_day = np.mean(res_i[:, 5])
      print('{:<9}   {:<6}    {:<6.2f}      {:<8.2f}    {:<8.2f}'.format(
        i, count, count / res.shape[0], avg_lower_size, avg_end_day))

  print('---------------------------------------------------------')


if __name__ == '__main__':
  for f in range(5, 20):
    for c in range(int(np.floor(1.5  *np.sqrt(f))), int(np.ceil(35*np.sqrt(f)))):
      main(n=10000, f=f, c=c, n_trials=1000)

