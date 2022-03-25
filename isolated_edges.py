import numpy as np
import math, sys, time, os, glob
from datetime import timedelta
from random_graph import RandomGraph

RED = 1
BLUE = -1


def print_progress_bar(start_time, n_trials_done, n_trials_total, count):
  # print progress bar
  sys.stdout.write(
    '\rCompleted: {}/{} trials. Count: {}. Elapsed: {}'.format(
      n_trials_done, n_trials_total, count,
      timedelta(seconds=time.time() - start_time)))
  sys.stdout.flush()


def count_isolated_edges_rb(edges, colors):
  neighborhood_of_reds = edges[colors == RED]
  degrees_of_reds = np.sum(neighborhood_of_reds, axis=1)
  temp1 = neighborhood_of_reds[degrees_of_reds == 1]  # Reds who have only 1 neighbor
  temp2 = temp1[temp1[:, colors == BLUE].any(axis=1)]  # filter out Reds whose only neighbor is Blue
  temp3 = np.dot(edges, temp2.T).sum(axis=0)  # degree of the only Blue neighbor of each such Red
  return np.count_nonzero(temp3 == 1)


def check_count_isolated_edges_rb():
  g = RandomGraph(6)
  g.set_color(0)
  g.generate(0.1)
  edges = np.floor(g.edges / 2).astype(int)
  print(edges)
  print(g.colors)
  print(count_isolated_edges_rb(edges, g.colors))


def trial(n, d, c=0):
  g = RandomGraph(n)
  g.set_color(c)
  g.generate(d / n)
  edges = np.floor(g.edges / 2).astype(int)
  return count_isolated_edges_rb(edges, g.colors)


def main(n=10000, d=1, c=0, n_trials=1000):
  print('Proceeding to run n_trials = {} trials on random graph G(n, d/n) '
        'for n = {}, d = {}.'.format(n_trials, n, d))
  print('Advantage c = {}. Starting with {} Reds and {} Blues.'.format(
    c, math.ceil(n / 2 + c), math.floor(n / 2 - c)))
  print()
  isolated_edges_occurences = 0
  start = time.time()
  for i in range(n_trials):
    n_isolated_edges = trial(n, d, c=c)
    isolated_edges_occurences += n_isolated_edges > 0
    print_progress_bar(start, i + 1, n_trials, isolated_edges_occurences)
  print()
  print('Isolated edges between R and B occur {} out of {} times ({}%)'.format(
    isolated_edges_occurences, n_trials, round(isolated_edges_occurences / n_trials * 100, 2)
  ))


if __name__ == '__main__':
  main(n=10000, d=1, c=0, n_trials=100)
