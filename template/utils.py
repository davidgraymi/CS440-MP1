'''
This file contains utilities for each State space
- for EightPuzzle we provide 
    - a function read_puzzle to read a puzzle file
- for WordLadder we provide
    - a function to read word ladders
    - a function to check if a word is English (against a dictionary)
    - a function to compute the Levenshtein distance between two words
- for GridState we provide
    - a class for computing the Minimum Spanning Tree
'''

# EightPuzzle ------------------------------------------------------------------------------------

def read_puzzle(filename):
    with open(filename, "r") as file:
        all_grids = []
        for line in file:
            grid = [[]]
            for c in line.strip():
                if len(grid[-1])==3:
                    grid.append([])
                intc = int(c)
                if intc == 0:
                    zero_loc = [len(grid)-1, len(grid[-1])]
                grid[-1].append(intc)
            all_grids.append([grid, zero_loc])
        return all_grids

def draw_eight_puzzle_path(path):
    """
    Draw an EightPuzzle path as a sequence of 3x3 boards.

    Each board except the last shows the next tile move with a blue arrow.
    """
    import math
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if len(path) == 0:
        print("No EightPuzzle path to draw")
        return

    grids = [state.state if hasattr(state, "state") else state for state in path]
    max_cols = 7
    cols = min(max_cols, len(grids))
    rows = math.ceil(len(grids) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.65 * cols, 1.9 * rows), squeeze=False)

    for step, ax in enumerate(axes.flat):
        ax.axis("off")
        if step >= len(grids):
            continue

        grid = grids[step]
        next_grid = grids[step + 1] if step + 1 < len(grids) else None
        blank = next_blank = None
        if next_grid is not None:
            for row in range(3):
                for col in range(3):
                    if grid[row][col] == 0:
                        blank = (row, col)
                    if next_grid[row][col] == 0:
                        next_blank = (row, col)

        moved_from = moved_to = None
        if blank is not None and next_blank is not None and blank != next_blank:
            moved_from = next_blank
            moved_to = blank

        ax.set_xlim(0, 3)
        ax.set_ylim(3, 0)
        ax.set_aspect("equal")
        ax.set_title("start" if step == 0 else f"move {step}", fontsize=9)

        for row in range(3):
            for col in range(3):
                tile = grid[row][col]
                facecolor = "whitesmoke" if tile == 0 else "white"
                edgecolor = "tab:blue" if moved_from == (row, col) else "black"
                linewidth = 2.0 if moved_from == (row, col) else 1.0
                ax.add_patch(Rectangle((col, row), 1, 1, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
                                       zorder=2 if moved_from == (row, col) else 1))
                if tile != 0:
                    ax.text(col + 0.5, row + 0.5, str(tile), ha="center", va="center", fontsize=14, weight="bold")

        if moved_from is not None and moved_to is not None:
            from_row, from_col = moved_from
            to_row, to_col = moved_to
            ax.annotate(
                "",
                xy=(to_col + 0.5, to_row + 0.5),
                xytext=(from_col + 0.5, from_row + 0.5),
                arrowprops={"arrowstyle": "->", "color": "tab:blue", "linewidth": 2.4, "shrinkA": 10, "shrinkB": 10},
            )

    fig.suptitle("EightPuzzle solution path", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()

# WordLadder ------------------------------------------------------------------------------------

def read_word_ladders():
    with open("data/word_ladder/ladder_problems.txt", "r") as file:
        all_pairs = []
        for line in file:
            words = line.split()
            if len(words) > 0:
                all_pairs.append(words)
    return all_pairs
    
with open("data/word_ladder/wiki-100k.txt", "rb") as word_file:
    english_words = set(word.strip().decode("utf-8").lower() for word in word_file if word[0] !="#")

def is_english_word(word):
    return word.lower() in english_words

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

# GridState ------------------------------------------------------------------------------------

# Compute the weight of a Minimum Spanning Tree over the objectives. We treat
# the objectives as nodes in a complete graph whose edge costs come from distance.
def compute_mst_cost(objectives, distance) -> float:
    # Each value is its node's parent; None marks the root of a component.
    elements = {key: None for key in objectives}

    # Find a component's root and compress the path for later lookups.
    def resolve(key):
        path = []
        while elements[key] is not None:
            path.append(key)
            key = elements[key]
        for item in path:
            elements[item] = key
        return key

    # Kruskal's algorithm adds the shortest edges that join different components.
    weight = 0
    edges = sorted(
        (distance(a, b), a, b)
        for a in objectives for b in objectives if a < b
    )
    for edge_cost, a, b in edges:
        root_a, root_b = resolve(a), resolve(b)
        if root_a != root_b:
            elements[root_b] = root_a
            weight += edge_cost
    return weight
