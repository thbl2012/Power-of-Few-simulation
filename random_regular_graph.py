import numpy as np


def get_uv_color_day1(n_red, n_blue, d, color_u=0, color_v=0):
  u = 0 if color_u else n_red
  v = 1 if color_v else n_red + 1
  n = n_red + n_blue
  neighbors_u = np.random.choice(
    np.concatenate((np.arange(u), np.arange(u+1, n))), size=d, replace=False
  )
  neighbors_v = np.random.choice(
    np.concatenate((np.arange(v), np.arange(v+1, n))), size=d, replace=False
  )
  dif_u = np.sum(neighbors_u < n_red) * 2 - d + color_u / 2
  dif_v = np.sum(neighbors_v < n_red) * 2 - d + color_v / 2
  return (dif_u > 0), (dif_v > 0)


def count_red_uv(n_red, n_blue, d, n_trials, color_u=0, color_v=0):
  red_count_u = 0
  red_count_v = 0
  red_count_both = 0
  for i in range(n_trials):
    color_u1, color_v1 = get_uv_color_day1(n_red, n_blue, d, color_u=color_u, color_v=color_v)
    red_count_u += color_u1
    red_count_v += color_v1
    red_count_both += (color_u1 and color_v1)
  return red_count_u, red_count_v, red_count_both


if __name__ == '__main__':
  n = 5000
  t = 100
  d = 5000
  n_trials = 10000
  color_u = 1
  color_v = 1
  n_red = n + t
  n_blue = n
  print('Running {} trials on {}-regular graph with {} nodes, {} Red and {} Blue'.format(
    n_trials, d, 2*n + t, n_red, n_blue
  ))
  print('C_0(u) = {}, C_0(v) = {}'.format(color_u, color_v))
  red_u, red_v, red_both = count_red_uv(
    n_red=n_red, n_blue=n_blue, d=d, n_trials=n_trials,
    color_u=color_u, color_v=color_v,
  )
  print('Count(u is red) = {}/{}. P(u is red) = {}'.format(red_u, n_trials, red_u / n_trials))
  print('Count(v is red) = {}/{}. P(v is red) = {}'.format(red_v, n_trials, red_v / n_trials))
  print('Count(uv are red) = {}/{}. P(uv are red) = {}'.format(red_both, n_trials, red_both / n_trials))
  print('Cov(C_1(u), C_1(v)) = {}'.format(red_both / n_trials - (red_u / n_trials) * (red_v / n_trials)))
