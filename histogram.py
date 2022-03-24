import numpy as np, os, glob
from matplotlib import pyplot as plt
from first_day import DATADIR, MAX_DAYS


def histogram(data, title='', bin_length=1):
  bins, edges, _ = plt.hist(data,
                            bins=np.arange(start=np.min(data),
                                           stop=np.max(data) + 1,
                                           step=bin_length),
                            density=True)
  plt.title(title)
  plt.show()
  return bins, edges


def get_data(dataname):
  return np.concatenate([np.load(fname) for fname in
                         glob.glob('{}/{}/*.npy'.format(DATADIR, dataname))], axis=0)


def blue_win_with_adv(dataname, n=0):
  data = get_data(dataname)
  if n == 0:
    n = int(dataname.split('_')[0])
  data_1 = data[data[:, 1] > n / 2]
  data_1_win = data_1[data_1[:, MAX_DAYS] == n]
  data_win = data[data[:, MAX_DAYS] == n]
  print('Probability Blue advantage day 1: {}/{} = {}'.format(
    data_1.shape[0], data.shape[0], data_1.shape[0] / data.shape[0]))
  print('Probability Blue win: {}/{} = {}'.format(
    data_win.shape[0], data.shape[0], data_win.shape[0] / data.shape[0]))
  print('Probability Blue win given advantage day 1: {}/{} = {}'.format(
    data_1_win.shape[0], data_1.shape[0], data_1_win.shape[0] / data_1.shape[0]))
  print('Probability Blue advantage day 1 given win: {}/{} = {}'.format(
    data_1_win.shape[0], data_win.shape[0], data_1_win.shape[0] / data_win.shape[0]))


def days_1_2(c):
  n = 3000
  day = 1
  data = get_data('{}_half_{}'.format(n, c))
  # data = data[data[:, MAX_DAYS] == n]
  data = data[:, day]
  print(data.shape)
  mean = np.mean(data)
  std = np.std(data)
  print('Mean: {}, Std: {}, Std/sqrt(n): {}, (n/2 - mean) / sqrt(n): {}'.format(
    mean, std,  std / np.sqrt(n), (n / 2 - mean) / np.sqrt(n)))
  bins, edges = histogram(data,
                          # title='G({}, 1/2) with c = {} advantage. '
                          #       'Distribution of number of Blues in Day {}'.format(n, c, day),
                          bin_length=1)
  # for i in range(bins.shape[0]):
  #   if bins[i] > 0:
  #     print('{} - {}'.format(edges[i], edges[i + 1]))
  print(bins.shape)
  plt.plot(edges[1:], np.log(bins))
  plt.show()


def blue_win_prob_day1(c):
  n = 3000
  blue_win_with_adv('{}_half_{}'.format(n, c))


def blue_win_prob_day1_max_deg(c):
  n = 10000
  blue_win_with_adv('max_deg_{}_half_{}'.format(n, c), n=10000)


def dist_compare_between_adv(day):
  n = 3000
  bins_arr = []
  edges_arr = []
  for c in range(1, 6):
    data = get_data('{}_half_{}'.format(n, c))[:, day]
    data = data - np.mean(data)
    bins, edges, _ = plt.hist(
      data, bins=np.arange(start=np.min(data), stop=np.max(data) + 1, step=3), density=True)
    bins_arr.append(bins)
    edges_arr.append(edges[1:])
  plt.close()
  for c in range(1, 6):
    plt.plot(edges_arr[c-1], bins_arr[c-1], label='c = {}'.format(c))
  # sigma = 0.5 * np.sqrt(n)
  # x = np.arange(start=-120, stop=120, step=3)
  # y = np.exp(- x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
  # plt.plot(x, y, label='gaussian', color='k')
  plt.legend(prop={'size': 15})
  plt.show()


def day2_log():
  n = 3000
  day = 2
  c = 1
  data = get_data('{}_half_{}'.format(n, c))[:, day]
  print(data.shape)
  mean = np.mean(data)
  std = np.std(data)

  bins, edges, _ = plt.hist(
    data, bins=np.arange(start=np.min(data), stop=np.max(data) + 1, step=10),
    density=True, cumulative=True)
  plt.show()

  plt.plot(np.log(1 - bins))
  plt.show()


def day1_critical_coef_by_c():
  n = 3000
  day = 1
  x = np.arange(start=0, stop=6, step=1.)
  y = np.zeros(6, dtype=float)
  for c in range(1, 6):
    data = get_data('{}_half_{}'.format(n, c))
    # data = data[data[:, MAX_DAYS] == n]
    data = data[:, day]
    y[c] = (n / 2 - np.mean(data)) / np.sqrt(n)
    print('For c = {}: (n/2 - mean) / (c * sqrt(n)) = {}'.format(c, y[c] / c))
  plt.plot(x, y, marker='.', markersize=20)
  plt.xlabel('c', fontsize=20)
  plt.ylabel('$\widetilde{d}$', fontsize=20)
  plt.show()


def day1_critical_coef_by_p(*clist):
  n = 3000
  day = 1
  m = 50
  x = np.arange(start=1, stop=50, step=1.) / m
  y = np.zeros(m-1, dtype=float)
  for c in clist:
    for i in range(1, m):
      data = get_data('{}_{}div{}_{}'.format(n, i, m, c))
      # data = data[data[:, MAX_DAYS] == n]
      data = data[:, day]
      y[i - 1] = (n / 2 - np.mean(data)) / np.sqrt(n)
      # print('For c = 1, p = {}: (n/2 - mean) / (c * sqrt(n)) = {}'.format(i / m, y[i - 1] / c))
    # y = y[1:] - y[:-1]
    y = np.log(y)
    plt.plot(x, y, marker='.', markersize=5, label='c = {}'.format(c))
  plt.xlabel('p', fontsize=20)
  plt.ylabel('$\widetilde{d}$', fontsize=20)
  plt.legend()
  plt.show()


def day1_variance_by_p(*clist):
  n = 3000
  day = 1
  m = 50
  x = np.arange(start=1, stop=50, step=1.) / m
  y = np.zeros(m-1, dtype=float)
  for c in clist:
    for i in range(1, m):
      data = get_data('{}_{}div{}_{}'.format(n, i, m, c))
      # data = data[data[:, MAX_DAYS] == n]
      data = data[:, day]
      y[i - 1] = np.var(data) / n
      # print('For c = 1, p = {}: (n/2 - mean) / (c * sqrt(n)) = {}'.format(i / m, y[i - 1] / c))
    plt.plot(x, y, marker='.', markersize=5, label='c = {}'.format(c))
  plt.xlabel('p', fontsize=20)
  plt.ylabel('$\widetilde{d}$', fontsize=20)
  plt.legend()
  plt.show()


if __name__ == '__main__':
  # blue_win_prob_day1_max_deg(1)
  # day1_critical_coef_by_p(1)
  day1_variance_by_p(1, 2, 3, 4)
