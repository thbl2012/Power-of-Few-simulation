import matplotlib.backend_bases
import matplotlib.pyplot as plt
import numpy as np

# this is the data structure is used by NavigationToolbar2
# to switch between different pans.  We'll make the figure's
# toolbar hold a proxy to such an object

from matplotlib.cbook import Stack


class StackProxy:
  def __init__(self, stack):
    self._stack = stack

  def __call__(self):
    return self._stack.__call__()

  def __len__(self):
    return self._stack.__len__()

  def __getitem__(self, ind):
    return self._stack.__getitem__(ind)

  def nop(self):
    pass

  # prevent modifying the stack
  def __getattribute__(self, name):
    if name == '_stack':
      return object.__getattribute__(self, '_stack')
    if name in ['push', 'clear', 'remove']:
      return object.__getattribute__(self, 'nop')
    else:
      return object.__getattribute__(self._stack, name)


stack = Stack()

for data in [[np.random.random(10), np.random.random(10)] for _ in range(5)]:
  stack.push(data)


def redraw(*args):
  plt.clf()
  plt.scatter(*stack())  # stack() returns the currently pointed to item
  plt.gcf().canvas.draw()


def main0():
  fig = plt.gcf()
  toolbar = fig.canvas.toolbar
  toolbar._update_view = redraw.__get__(toolbar)
  stackProxy = StackProxy(stack)
  toolbar._nav_stack = stackProxy
  redraw()
  plt.show()


def f0(n):
  loga = np.log(np.arange(1, n+1))  # loga[k-1] = log(k)
  loga_fact = np.cumsum(loga)  # loga_fact[k-1] = log(1) + ... + log(k)
  loga_pow = np.arange(n) * loga  # loga_pow[k-1] = (k-1)log(k)
  f = loga_pow - loga_fact  # f[k-1] = log(k^(k-1)/k!)
  exps = f[:-1] + f[-2::-1] - f[n-1]
  return np.sum(np.exp(exps))



from scipy.stats import binom


def trans_mat(n, p):
  one_to_n = np.arange(1, n, dtype=int)
  a1 = binom.pmf(n=n, p=p, k=one_to_n)
  A1 = np.tile(a1, (n-1, 1))
  temp1 = np.tril(np.tile(one_to_n[:, None], (1, n-1)), -1)
  A2_ns = np.fliplr(temp1) + np.flipud(temp1)
  A2_pows = n - A2_ns
  A2_ks = A2_pows.T
  temp2 = np.tri(n-1, k=-1)
  A2_ps = p * np.flipud(temp2) + (1 - p) * np.fliplr(temp2)
  A2_exps = (1 - p) * np.flipud(temp2) + p * np.fliplr(temp2)
  A2 = A2_exps ** A2_pows * binom.pmf(n=A2_ns, p=A2_ps, k=A2_ks)
  return A1 - A2


def left_right_mat(n, p):
  temp1 = np.arange(1, n, dtype=int)[:, None]
  p2 = (1 - p) ** temp1
  p1 = p ** np.flipud(temp1)
  prod = p1 * p2
  return p2 - prod, p1 - prod


if __name__ == '__main__':
  n = 100
  p = 0.51
  m = trans_mat(n, p)
  # m1 = np.linalg.inv(np.identity(n-1) - m) - np.identity(n-1)
  l, r = left_right_mat(n, p)
  xl = np.linalg.solve(np.identity(n-1) - m, l)
  # xr = np.linalg.solve(m, r)
  print(xl)
  # print()
  # print(xr)
  # plt.imshow(m1)
  # plt.colorbar()
  # plt.plot(m1[:, 0])
  # plt.plot(m1[:, 20])
  # plt.plot(m1[:, 40])
  # plt.plot(m1[:, 60])
  # plt.plot(m1[:, 80])
  # plt.show()

