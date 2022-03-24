from random_graph import RandomGraph
import numpy as np, math, sys, os, glob, time


MAX_DAYS = 6
SAVE_PERIOD = 1000
DATADIR = 'voting_data'
MAX_DIGITS = 6
SECS_PER_HR = 3600


# Simulate the election process on G(n, p) with advantage c
# Only proceeds if Blue gains advantage after Day 1
# Returns the winner and end day if ends within 20 days
# Otherwise return "Inconclusive"
def trial(n, p, c, initial_coloring=None):
  g = RandomGraph(n)
  if initial_coloring is None:
    g.set_color(c)
    g.generate(p)
  else:
    g.generate(p)
    g.set_color(c, coloring=initial_coloring)
  results = np.empty(MAX_DAYS + 1, dtype=np.int)
  results[0] = g.count()[1]

  for day in range(1, MAX_DAYS + 1):
    g.transition()
    n_reds, n_blues = g.count()
    results[day] = n_blues
    if n_blues == 0 or n_reds == 0:
      results[day + 1:] = n_blues
      break
  return results


def batch_sim(n, p, c, n_trials, name, initial_coloring=None, verbose=True):
  n_batches = n_trials // SAVE_PERIOD
  n_trials = n_batches * SAVE_PERIOD
  os.makedirs('{}/{}'.format(DATADIR, name), exist_ok=True)
  batch_list = [int(fname[-MAX_DIGITS-4:-4]) for fname in glob.glob('{}/{}/*.npy'.format(DATADIR, name))]
  first_batch = max(batch_list) + 1 if batch_list else 0

  print('Running {} trials (rounded to multiple of {})'
        'on G({}, {}).'.format(n_trials, SAVE_PERIOD, n, p))
  print('Advantage c = {}. Starting with {} Reds and {} Blues.'.format(
      c, math.ceil(n / 2 + c), math.floor(n / 2 - c)))
  print()

  # run n_trials trials
  start = time.time()
  for batch in range(n_batches):
    records = np.empty((SAVE_PERIOD, MAX_DAYS + 1), dtype=np.int)
    batch_start = batch * SAVE_PERIOD + 1
    for i in range(SAVE_PERIOD):
      records[i] = trial(n, p, c, initial_coloring=initial_coloring)
      # print progress bar
      if verbose:
        sys.stdout.write('\rCompleted: {}/{} trials ({}%). Time elapsed: {} hours'.format(
          batch_start + i, n_trials, math.floor((batch_start + i) / n_trials * 100),
          (time.time() - start) / SECS_PER_HR))
        sys.stdout.flush()
    np.save('{}/{}/{:0{}}.npy'.format(DATADIR, name, batch + first_batch, MAX_DIGITS), records)
  print()


def main():
  n = 10000
  p = 0.5
  c = 0
  n_trials = SAVE_PERIOD * 10
  name = 'max_deg_{}_half_{}'.format(n, c)
  batch_sim(n, p, c, n_trials, name, initial_coloring='max_deg', verbose=True)


def main_multiple_p(a):
  n = 3000
  p = a / 50
  c = 4
  n_trials = SAVE_PERIOD * 10
  name = '{}_{}div50_{}'.format(n, a, c)
  batch_sim(n, p, c, n_trials, name, initial_coloring='max_deg', verbose=True)


if __name__ == '__main__':
  for i in range(1, 50):
    main_multiple_p(i)
