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


def compare(n, p):
  a1 = binom.cdf(np.arange(0, n + 0.5, 0.5), n, p)
  a2 = binom.cdf(np.arange(0, 2*n + 1, 1), 2*n, p)
  plt.plot(np.log(a1))
  plt.plot(np.log(a2))
  plt.show()


if __name__ == '__main__':
  compare(100, 0.3)

