import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle

class Maze:
    def __init__(self, file_path, allow_waiting = False):
        """
        Creates a maze instance given a `file_path` to a text file containing the ASCII representation of a maze.
        Key:
            - Walls are represented by %
            - Open paths by spaces
            - Starts by capital letters (at most one of each letter) 
            - Goals (waypoints) by lowercase letters matching one of the starts
        
        If `allow_waiting` is True, the agent can stay in place (not move) as a valid action.
        """
        self.file_path = file_path
        self.allow_waiting = allow_waiting
        with open(file_path) as file:
            lines = tuple(line.strip() for line in file.readlines() if line)
        
        wall_char = '%'
        free_char = ' '
        height = len(lines)
        width = len(lines[0])
        
        # check that we have a rectangular grid
        if any(len(line) != width for line in lines):
            raise ValueError(f'(maze {file_path}): all maze rows must be the same length')
        
        # read the maze from the file into a numpy array and store starts and goals
        self.grid = np.zeros((height, width))
        self.starts = {}
        self.goals = {}
        for i in range(height):
            for j in range(width):
                cur_char = lines[i][j]
                if cur_char == wall_char:
                    self.grid[i, j] = 1
                elif cur_char.isupper():
                    if cur_char.lower() in self.starts:
                        raise ValueError(f'(maze {file_path}): starts must be unique')
                    self.starts[cur_char.lower()] = (i, j)
                elif cur_char.islower():
                    if cur_char not in self.goals:
                        self.goals[cur_char] = ()
                    self.goals[cur_char] += ((i, j),)
                elif cur_char != free_char:
                    raise ValueError(f'(maze {file_path}): invalid character {cur_char}')        
                
        # check that every start has a corresponding goal
        for start in self.starts:
            if start not in self.goals:
                raise ValueError(f'(maze {file_path}): start {start} has no corresponding goal')

        # check that border contains walls
        if np.any(self.grid[0, :]==0) or\
            np.any(self.grid[-1, :]==0) or\
                np.any(self.grid[:, 0]==0) or\
                    np.any(self.grid[:, -1]==0):
            raise ValueError(f'(maze {file_path}): border of maze must be walls')

        # This is a helper to track number of times we call self.is_free(i, j)
        self.num_states_validated = 0
    
    def in_bounds(self, i, j):
        """Check if cell (i,j) is in bounds"""
        return 0 <= i < self.grid.shape[0] and 0 <= j < self.grid.shape[1]
    
    def is_free(self, i, j):
        """Check if in bounds cell (i,j) is free - not a wall"""
        self.num_states_validated += 1
        return self.grid[i, j] != 1

    def neighboring_cells(self, i, j):
        """Returns the in-bounds cells neighboring the given row,col."""
        possible_moves = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
        if self.allow_waiting:
            possible_moves.append((i, j))
        return tuple(x for x in possible_moves if self.in_bounds(*x))

    def valid_path(self, path):
        # validate type and shape 
        if len(path) == 0:
            print(f'Invalid path: path must contain at least one element')
            return False
        if not all(len(vertex) == 2 for vertex in path):
            print(f'Invalid path: each path element must be a two-element sequence')
            return False
        
        # normalize path in case student used an element type that is not `tuple` 
        path = tuple(map(tuple, path))

        # check if path is contiguous
        for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
            d = sum(abs(b_ - a_) for a_, b_ in zip(a, b)) 
            if d > 1:
                print(f'Invalid path: path vertex {i} {a} is too far from consecutive path vertex {i + 1} {b}')
                return False
            if d == 0 and not self.allow_waiting:
                print(f'Invalid path: path vertex {i} {a} is the same as consecutive path vertex {i + 1} {b}, and waiting is not allowed')
                return False

        # check if path is navigable 
        for i, x in enumerate(path):
            if not self.in_bounds(*x) or not self.is_free(*x):
                print(f'Invalid path: path vertex {i} {x} is not a navigable maze cell')
                return False
        
        # the path must start at a start location
        if path[0] not in self.starts.values():
            print(f'Invalid path: first path vertex {path[0]} must be a start location')
            return False
        
        # get the goal associated with the path start
        path_goals = None
        for c, start in self.starts.items(): # works for multi-agent (MP2)
            if start == path[0]:
                path_goals = self.goals[c]                
                break
        
        # check if the path ends at a goal 
        if path[-1] not in path_goals:
            print(f'Invalid path: last path vertex {path[-1]} must be a goal')
            return False
        
        # check for unnecessary path segments (looping back to a previous location without visiting a waypoint)
        if not self.allow_waiting:
            indices = {}
            for i, x in enumerate(path):
                if x in indices:
                    if all(x not in path_goals for x in path[indices[x] : i]):
                        print(f'Bad path: path segment [{indices[x]} : {i}] contains no waypoints but loops back to a previous location')
                        return False
                indices[x] = i 
        
        # check if path contains all waypoints 
        for goal in path_goals:
            if goal not in path:
                print(f'Bad path: path must contain all waypoints')
                return False
        
        return True

    def draw_maze(self, path=None, save=None, show=True):
        """Draw a simple maze visualization, optionally with a path"""
        if path is None:
            path = []
        # in case the user calls with a sequence of GridState-like objects...
        path = [item.state if hasattr(item, "state") else item for item in path]
        # verify type and shape of path elements, raise ValueError if not valid
        for item in path:
            is_valid_loc = (
                hasattr(item, "__len__") and len(item) == 2
                and all(isinstance(v, (int, np.integer)) for v in item)
            )
            if not is_valid_loc:
                raise ValueError("draw_maze only supports single-agent paths - a sequence of 2-integer tuples")
        path = tuple(map(tuple, path))  # normalize to tuple of tuples

        # first set up the figure and axes with a black and white grid for the maze
        height, width = self.grid.shape
        fig, ax = plt.subplots(layout="constrained")
        ax.imshow(self.grid, cmap=ListedColormap(["white", "black"]), vmin=0, vmax=1)
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.grid(which="minor", color="lightgray", linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_aspect("equal") # ensure square cells even if the figure is resized
        ax.set_title(self.file_path)

        # show the path as a heatmap of distance from start, ignore anything not on path
        heatmap = np.full(self.grid.shape, np.nan)
        for step, (row, col) in enumerate(path):
            heatmap[row, col] = step
        if np.any(~np.isnan(heatmap)):
            path_cmap = plt.cm.viridis.copy()
            path_cmap.set_bad(alpha=0)
            img = ax.imshow(np.ma.masked_invalid(heatmap), cmap=path_cmap, interpolation="none", alpha=0.75)
            cbar = fig.colorbar(img, ax=ax, location="bottom", fraction=0.05, pad=0.1)
            cbar.set_label("distance from start")

        # for drawing goals it's helpful to see explicitly the order they are visited
        goal_locs = {tuple(loc) for positions in self.goals.values() for loc in positions}
        goal_order = {}
        if len(goal_locs) > 1:
            for loc in path: # only track goals that are visited
                if loc in goal_locs and loc not in goal_order:
                    goal_order[loc] = len(goal_order) + 1

        # draw the starts as squares
        marker_size = 0.75
        for idx, (row, col) in enumerate(self.starts.values()):
            ax.add_patch(Rectangle((col - marker_size/2, row - marker_size/2), marker_size, marker_size,
                         facecolor="tab:blue", edgecolor="white", label="start" if idx == 0 else None, zorder=4))
        # draw the goals as circles, and if there are multiple goals, label them with the order they are visited
        goal_idx = 0
        for goal_positions in self.goals.values():
            for row, col in goal_positions:
                ax.add_patch(Circle((col, row), marker_size/2, facecolor="tab:green", edgecolor="white",
                             label="goal" if goal_idx == 0 else None, zorder=4))
                goal_idx += 1
                # only label goal order for visited goals
                if (row, col) in goal_order:
                    ax.text(col, row, str(goal_order[(row, col)]), color="white",
                            ha="center", va="center", fontsize="xx-small", weight="bold", zorder=5)

        fig.legend(loc="outside lower center", ncol=2, frameon=False)

        if save is not None:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, ax
