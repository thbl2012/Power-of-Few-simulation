import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.cbook import Stack
from random_graph import Graph


INDICATOR_PARENT = 2
INDICATOR_CHILD = 1
COLOR_RED = 1
COLOR_BLUE = -1
COLOR_CODES = {COLOR_RED: 'red', COLOR_BLUE: 'blue'}


class RandomTree(Graph):
  def generate(self):
    n = self.size
    self.edges = np.zeros((n, n), dtype=np.int32)
    # Initial partition: (tree, remains) = ([0], all nodes > 0)
    curr_partition = np.arange(0, n)
    curr_tree_size = 1
    # Each step, choose a random node in tree and a random node in remains and connect them,
    # then add the connected node to tree by swapping it with the node at index [curr_tree_size]
    for curr_tree_size in range(1, n):
      i = np.random.choice(curr_partition[:curr_tree_size], size=1)
      index = np.random.randint(curr_tree_size, n)
      j = curr_partition[index]
      self.edges[i, j] = 2
      self.edges[j ,i] = 2
      curr_partition[curr_tree_size], curr_partition[index] = curr_partition[index], curr_partition[curr_tree_size]
    np.fill_diagonal(self.edges, 1)


# Get the width (= number of leaves) at this rooted tree and each subtree and save in result
# Also update parent-child relation according to this rooting
def get_width_at_root(edges, root, result):
  children = (edges[root] == INDICATOR_CHILD).nonzero()[0]
  if len(children) == 0:  # If a leaf is reached
    return 0  # Leaf has width 0
  for child in children:
    # Update parent-child relation
    edges[child, root] = INDICATOR_PARENT
    # If child is a leaf, it reports 1 back to its parent instead
    result[root] += max(1, get_width_at_root(edges, child, result))
  return result[root]


def child_position_in_mid_arc(x_root, y_root, low, high, length=1):
  mid = (low + high) / 2
  return round(x_root + length * math.cos(mid), 3), round(y_root + length * math.sin(mid), 3)


# Returns list of nodes in the subtree from this root and list of edges to plot
def arrange_tree(edges, widths, root, edge_count, position=None,
                 low_angle=0, high_angle=2 * math.pi, edge_length=1,
                 nodes_x=None, nodes_y=None, edges_x=None, edges_y=None):
  if position is None:
    position = (0, 0)
  # Empty arrays to store result of arrange_tree
  n = len(widths)
  nodes_x = np.empty(n) if nodes_x is None else nodes_x
  nodes_y = np.empty(n) if nodes_y is None else nodes_y
  edges_x = np.empty((n, 2)) if edges_x is None else edges_x
  edges_y = np.empty((n, 2)) if edges_y is None else edges_y
  x_root, y_root = position
  nodes_x[root] = x_root
  nodes_y[root] = y_root
  children = (edges[root] == INDICATOR_CHILD).nonzero()[0]
  if len(children) != 0:
    arc = high_angle - low_angle
    low = low_angle
    for i, child in enumerate(children):
      high = low + arc * max(1, widths[child]) / widths[root]
      x_child, y_child = child_position_in_mid_arc(x_root, y_root, low, high, length=edge_length)
      edges_x[edge_count] = x_child, x_root
      edges_y[edge_count] = y_child, y_root
      edge_count, _, _, _, _ = arrange_tree(
        edges, widths, child, edge_count + 1, position=(x_child, y_child),
        low_angle=low, high_angle=high, edge_length=edge_length,
        nodes_x=nodes_x, nodes_y=nodes_y, edges_x=edges_x, edges_y=edges_y
      )
      low = high
  return edge_count, nodes_x, nodes_y, edges_x, edges_y


def draw_tree(edges, widths, colors, root, position=None,
              low_angle=0, high_angle=2*math.pi,
              edge_length=1, edge_width=2, name_ver_pos=0.5, name_hor_pos=0.5,
              node_radius=5, fontsize=20):
  n = len(widths)
  count, nodes_x, nodes_y, edges_x, edges_y = arrange_tree(
    edges, widths, root, 0, position=position,
    low_angle=low_angle, high_angle=high_angle, edge_length=edge_length)
  # Plot nodes using stored info
  scatters = plt.scatter(nodes_x, nodes_y, marker='o', s=node_radius**2, zorder=2,
                         color=[COLOR_CODES[colors[i]] for i in range(n)])
  for i in range(n):
    plt.annotate(str(i), (nodes_x[i], nodes_y[i]),
                 (nodes_x[i] + name_hor_pos, nodes_y[i] + name_ver_pos),
                 ha='center', va='center', weight='heavy',
                 fontsize=fontsize, color='white')
  # Plot edges using stored info
  for i in range(count):
    plt.plot(edges_x[i], edges_y[i], linewidth=edge_width, color='k', zorder=1)
  return scatters


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


def draw_tree_transition(edges, widths, colorstack, root, position=None,
                         low_angle=0, high_angle=2*math.pi,
                         edge_length=1, edge_width=2, name_ver_pos=0.5, name_hor_pos=0.5,
                         node_radius=5, fontsize=20):
  sc = draw_tree(edges, widths, colorstack(), root, position=position,
                 low_angle=low_angle, high_angle=high_angle, edge_length=edge_length,
                 edge_width=edge_width, name_ver_pos=name_ver_pos, name_hor_pos=name_hor_pos,
                 node_radius=node_radius, fontsize=fontsize)

  def redraw(*args):
    # plt.clf()
    sc.set_color([COLOR_CODES[c] for c in colorstack()])
    plt.gcf().canvas.draw()

  fig = plt.gcf()
  toolbar = fig.canvas.toolbar
  toolbar._update_view = redraw.__get__(toolbar)
  stackProxy = StackProxy(colorstack)
  toolbar._nav_stack = stackProxy


def check_tree_transition(n=10, root=0, n_days=5):
  tree = RandomTree(size=n)
  tree.generate()
  tree.set_color(0)
  edges = np.floor(tree.edges / 2).astype(int)
  widths = np.zeros(n, dtype=int)
  get_width_at_root(edges, root, widths)
  colorstack = Stack()
  colorstack.push(tree.colors)
  for i in range(n_days):
    tree.transition()
    colorstack.push(tree.colors)
  draw_tree_transition(
    edges, widths, colorstack, root,
    edge_length=10, edge_width=2, name_ver_pos=0, name_hor_pos=0,
    node_radius=20, fontsize=12)
  plt.show()


def check_tree(n=10, root=0, n_days=5):
  tree = RandomTree(size=n)
  tree.generate()
  tree.set_color(0)
  edges = np.floor(tree.edges / 2).astype(int)
  widths = np.zeros(n, dtype=int)
  get_width_at_root(edges, root, widths)
  draw_tree(edges, widths, tree.colors, root,
            edge_length=10, edge_width=2, name_ver_pos=0, name_hor_pos=0,
            node_radius=20, fontsize=12)
  plt.show()
  for i in range(n_days):
    tree.transition()
    draw_tree(edges, widths, tree.colors, root,
              edge_length=10, edge_width=2, name_ver_pos=0, name_hor_pos=0,
              node_radius=20, fontsize=12)
    plt.show()


if __name__ == '__main__':
  check_tree_transition(n=30, n_days=10)


#
#
# OLD CODE

class TreeNode:
  def __init__(self, name: int, *children):
    self.name = name
    self.parent = None
    self.children = children
    for child in self.children:
      child.parent = self


def draw_tree_old(edges, widths, colors, root, position=None,
                  low_angle=0, high_angle=2 * math.pi,
                  edge_length=1, edge_width=2, name_ver_pos=0.5, name_hor_pos=0.5,
                  node_radius=5, fontsize=20):
  if position is None:
    position = (0, 0)
  x_root, y_root = position
  # plt.gca().set_aspect('equal', adjustable='box')
  plt.scatter([x_root], [y_root], marker='o',
              color=COLOR_CODES[colors[root]], s=node_radius**2, zorder=2)
  plt.annotate(str(root), (x_root, y_root), (x_root + name_hor_pos, y_root + name_ver_pos),
               ha='center', va='center', weight='heavy',
               fontsize=fontsize, color='white')
  # plt.text(x_root + name_hor_pos, y_root + name_ver_pos, str(root),
  #          color=COLOR_CODES[colors[root]], fontsize=fontsize)
  children = (edges[root] == INDICATOR_CHILD).nonzero()[0]
  if len(children) == 0:
    return
  arc = high_angle - low_angle
  low = low_angle
  for i, child in enumerate(children):
    high = low + arc * max(1, widths[child]) / widths[root]
    child_pos = child_position_in_mid_arc(x_root, y_root, low, high, length=edge_length)
    x_child, y_child = child_pos
    plt.plot([x_child, x_root], [y_child, y_root],
             linewidth=edge_width, color='k', zorder=1)
    draw_tree_old(edges, widths, colors, child, position=child_pos,
                  low_angle=low, high_angle=high,
                  edge_length=edge_length, edge_width=edge_width,
                  name_ver_pos=name_ver_pos, name_hor_pos=name_hor_pos,
                  node_radius=node_radius, fontsize=fontsize)
    low = high
