# Power-of-Few-simulation
Code for Power of Few simulations. The repo consists of utility files and experiment files.
The experiment files are to be run with a Python IDE or with command line.
Parameters for experiments are to be changed directly to the bottom lines in the experiement files.
Below is the list of experiments:

- `blues_by_day.py`: Runs many trials, each trial generates a random G(n, p) graph, assigns an initial coloring, then performs majority dynamics on the graph for a set a period of time. For each trial, the number of blues on each day is saved.

- `record_degrees_and_period.py`: Runs trials the same way as above, but the periodicity of the final state of each trial is recorded. Period 1 means the final state is stable. Period 2 means the process eventually switches between 2 colorings.

- `graph_visualization.py`: Runs trials on small graphs with a small number of days. The graph is drawn and shown with pyplot, with each day's coloring being visible with a mouse click. Currently trees and circular planar graphs are supported. To visualize trees, use the function `check_tree_transition`. To visualize circular graphs, use the function `check_circ_graph_transition`.

Please see the documentation strings in each file for the meaning of each function argument and how to run the main functions correctly.
