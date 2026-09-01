# annotations import allows us to specify that get_neighbors returns a list of AbstractState objects
from __future__ import annotations

# abstractmethod is a decorator indicating that a method is abstract and must be implemented by subclasses
from abc import ABC, abstractmethod
from itertools import count
import copy # you may want to use copy when creating neighbors for EightPuzzle...
from utils import is_english_word, levenshteinDistance, compute_mst_cost

class AbstractState(ABC):
    # This count increments every time a new AbstractState is created
    _tiebreak_count = count()

    def __init__(self, state, goal, dist_from_start=0, use_heuristic=True) -> None:
        self.state = state
        self.goal = goal
        # we tiebreak based on the order that the state was created/found
        self.tiebreak_idx = next(AbstractState._tiebreak_count)
        # dist_from_start is "g" in A*, i.e., f = g + h
        self.dist_from_start = dist_from_start
        self.use_heuristic = use_heuristic
        if use_heuristic:
            # NOTE: we only want to compute the heuristic once per state - 
            #       you should not call self.compute_heuristic anywhere else, just use self.h 
            self.h = self.compute_heuristic()
        else:
            self.h = 0

    # Return a list of AbstractState objects that can be reached from the current state
    @abstractmethod
    def get_neighbors(self) -> list[AbstractState]:
        pass
    
    # Return True if the state is the goal
    @abstractmethod
    def is_goal(self) -> bool:
        pass
    
    # Return a float estimating the cost to reach the goal from this state
    @abstractmethod
    def compute_heuristic(self) -> float:
        return 0
    
    # The "less than" method ensures that states are comparable.
    # self.dist_from_start is g, self.h is h, and self.tiebreak_idx is the tiebreaker.
    def __lt__(self, other : AbstractState) -> bool:
        f_self = self.dist_from_start + self.h
        f_other = other.dist_from_start + other.h

        if f_self < f_other:
            return True
        elif f_self == f_other:
            # Prefer smaller h value
            if self.h < other.h:
                return True
            elif self.h > other.h:
                return False
            # Fallback to created first
            return self.tiebreak_idx < other.tiebreak_idx
        else:
            return False

    # The "hash" method allows us to keep track of which states have been visited before in a dictionary
    # You should hash states based on self.state (and sometimes self.goal, if it can change)
    @abstractmethod
    def __hash__(self) -> int:
        pass
    # __eq__ gets called during hashing collisions, without it Python checks object equality
    @abstractmethod
    def __eq__(self, other) -> bool:
        pass
    
# WordLadder ------------------------------------------------------------------------------------------------

class WordLadderState(AbstractState):
    def __init__(self, state, goal, dist_from_start, use_heuristic, cost_per_letter):
        '''
        state: string of length n
        goal: string of length n
        dist_from_start: integer
        use_heuristic: boolean
        cost_per_letter: dictionary mapping letters to their cost, e.g., {'a': 1, 'b': 2, ...}
        '''
        # NOTE: AbstractState constructor does not take cost_per_letter
        super().__init__(state, goal, dist_from_start, use_heuristic)
        self.cost_per_letter = cost_per_letter
        
    # Each word can have many neighbors:
    #   Every letter in the word (self.state) can be replaced by every letter in the alphabet
    #   The resulting word must be a valid English word (i.e., in our dictionary)
    def get_neighbors(self) -> list[WordLadderState]:
        nbr_states = []
        for word_idx in range(len(self.state)):
            prefix = self.state[:word_idx]
            suffix = self.state[word_idx+1:]
            # 'a' = 97, 'z' = 97 + 25 = 122
            for c_idx in range(97, 97+26):
                c = chr(c_idx) # convert index to character
                # Replace the character at word_idx with c
                potential_nbr = prefix + c + suffix
                edge_cost = self.cost_per_letter[c]
                # If the resulting word is a valid english word, add it as a neighbor
                # NOTE: dist_from_start increases by edge_cost (this may not be 1!)
                # NOTE: This may generate the current word as a neighbor. The search code will ignore
                #       this self-loop because it has already visited the current word.
                if is_english_word(potential_nbr):
                    new_state = WordLadderState(
                        state=potential_nbr,
                        goal=self.goal, # stays the same!
                        dist_from_start=self.dist_from_start + edge_cost,
                        use_heuristic=self.use_heuristic, # stays the same!
                        cost_per_letter=self.cost_per_letter # stays the same!
                    )
                    nbr_states.append(new_state)
        return nbr_states

    # Checks if we reached the goal word with a simple string equality check
    def is_goal(self) -> bool:
        return self.state == self.goal
    
    # Strings are hashable, directly hash self.state
    def __hash__(self) -> int:
        return hash(self.state)
    def __eq__(self, other: AbstractState) -> bool:
        return self.state == other.state
    
    # The heuristic we use is the edit distance (Levenshtein) between our current word and the goal word
    # You may want to ask yourself: which constraint does this heuristic relax/remove from the original problem?
    def compute_heuristic(self) -> float:
        return levenshteinDistance(self.state, self.goal)
    
    # str and repr just make output more readable when you print out states
    def __str__(self):
        return self.state
    def __repr__(self):
        return self.state

# EightPuzzle ------------------------------------------------------------------------------------------------

def manhattan(a, b) -> float:
    return sum([abs(a[i] - b[i]) for i in range(len(a))])

class EightPuzzleState(AbstractState):
    def __init__(self, state, goal, dist_from_start, use_heuristic, zero_loc):
        '''
        state: 3x3 array of integers 0-8
        goal: 3x3 goal array, default is np.arange(9).reshape(3,3).tolist()
            NOTE: NOT a numpy array
        zero_loc: an additional helper argument indicating the 2d index of 0 in state, 
            you do not have to use it but we recommend setting it appropriately inside of get_neighbors
        '''
        # NOTE: AbstractState constructor does not take zero_loc
        super().__init__(state, goal, dist_from_start, use_heuristic)
        self.zero_loc = zero_loc
        self.print_one = False
    
    def get_neighbors(self) -> list[EightPuzzleState]:
        nbr_states = []
        # NOTE: There are *up to 4* possible neighbors and the order you add them matters for tiebreaking
        #   Please add them in the following order: [below, left, above, right], where for example "below" 
        #   corresponds to moving the empty tile down (moving the tile below the empty tile up)
        # Your code here ---------------

        if self.print_one:
            print(repr(self), "self")
        
        if self.zero_loc[0] < 2:
            new_zero_loc = (self.zero_loc[0] + 1, self.zero_loc[1])
            new_state = copy.deepcopy(self.state)
            new_state[self.zero_loc[0]][self.zero_loc[1]] = new_state[new_zero_loc[0]][new_zero_loc[1]]
            new_state[new_zero_loc[0]][new_zero_loc[1]] = 0
            below: EightPuzzleState = EightPuzzleState(
                new_state,
                self.goal,
                self.dist_from_start + 1, 
                self.use_heuristic,
                new_zero_loc
            )
            if self.print_one:
                print(repr(below), "below")
            nbr_states.append(below)

        if self.zero_loc[1] > 0:
            new_zero_loc = (self.zero_loc[0], self.zero_loc[1] - 1)
            new_state = copy.deepcopy(self.state)
            new_state[self.zero_loc[0]][self.zero_loc[1]] = new_state[new_zero_loc[0]][new_zero_loc[1]]
            new_state[new_zero_loc[0]][new_zero_loc[1]] = 0
            left: EightPuzzleState = EightPuzzleState(
                new_state,
                self.goal,
                self.dist_from_start + 1, 
                self.use_heuristic,
                new_zero_loc
            )
            if self.print_one:
                print(repr(left), "left")
            nbr_states.append(left)

        if self.zero_loc[0] > 0:
            new_zero_loc = (self.zero_loc[0] - 1, self.zero_loc[1])
            new_state = copy.deepcopy(self.state)
            new_state[self.zero_loc[0]][self.zero_loc[1]] = new_state[new_zero_loc[0]][new_zero_loc[1]]
            new_state[new_zero_loc[0]][new_zero_loc[1]] = 0
            above: EightPuzzleState = EightPuzzleState(
                new_state,
                self.goal,
                self.dist_from_start + 1, 
                self.use_heuristic,
                new_zero_loc
            )
            if self.print_one:
                print(repr(above), "above")
            nbr_states.append(above)

        if self.zero_loc[1] < 2:
            new_zero_loc = (self.zero_loc[0], self.zero_loc[1] + 1)
            new_state = copy.deepcopy(self.state)
            new_state[self.zero_loc[0]][self.zero_loc[1]] = new_state[new_zero_loc[0]][new_zero_loc[1]]
            new_state[new_zero_loc[0]][new_zero_loc[1]] = 0
            right: EightPuzzleState = EightPuzzleState(
                new_state,
                self.goal,
                self.dist_from_start + 1, 
                self.use_heuristic,
                new_zero_loc
            )
            if self.print_one:
                print(repr(right), "right")
            nbr_states.append(right)

        if self.print_one:
            self.print_one = False
        
        # ------------------------------
        return nbr_states

    # Checks if goal has been reached
    def is_goal(self) -> bool:
        # In python "==" performs deep list equality checking, so this works as desired
        return self.state == self.goal
    
    # Can't hash a list, so first flatten the 2d array and then turn into tuple
    def __hash__(self) -> int:
        return hash(tuple([item for sublist in self.state for item in sublist]))
    def __eq__(self, other) -> bool:
        return self.state == other.state
    
    def compute_heuristic(self) -> float:
        # print(repr(self))
        total = 0
        def find_position(grid, target):
            for r, row in enumerate(grid):
                for c, val in enumerate(row):
                    if val == target:
                        return (r, c)
            return None

        for row_index, row in enumerate(self.state):
            for col_index in range(len(row)):
                target = self.state[row_index][col_index]
                if target == 0:
                    continue

                start = (row_index, col_index)
                end = find_position(self.goal, target)
                man = manhattan(start, end)
                total += man
        # print(f'total distance is {total}')
        return total
    
    # NOTE: we emphasize for you that you do NOT need to touch the __lt__ method, 
    # you already implemented it in AbstractState
    
    # str and repr just make output more readable when you print out states
    def __str__(self):
        return str(self.state)
    def __repr__(self):
        return "\n---\n"+"\n".join([" ".join([str(r) for r in c]) for c in self.state])

# SingleGoalGridState --------------------------------------------------------------------------------

class SingleGoalGridState(AbstractState):
    # state: a length 2 tuple indicating the current location in the grid, e.g., (row, col)
    # goal: a length 2 tuple indicating the goal location, e.g., (row, col)
    # maze: Maze object for the problem. You can call maze.neighboring_cells(...) and maze.is_free(...)
    def __init__(self, state, goal, dist_from_start, use_heuristic, maze):
        self.maze = maze
        super().__init__(state, goal, dist_from_start, use_heuristic)
        
    def get_neighbors(self) -> list[SingleGoalGridState]:
        nbr_states = []
        # neighboring_cells is a tuple of tuples of neighboring locations, e.g., ((row1, col1), (row2, col2), ...)
        # Some of these cells may be walls. Use self.maze.is_free(...) to keep only valid locations.
        # feel free to look into maze.py for more details... we also recommend printing neighboring_cells to see what it looks like
        neighboring_cells = self.maze.neighboring_cells(*self.state)
        # TODO(V): fill this in
        # The distance from the start to a neighbor is always 1 more than the distance to the current state
        # Your code here ---------------
        for neighbor_cell in neighboring_cells:
            if self.maze.is_free(neighbor_cell[0], neighbor_cell[1]):
                neighbor_state = SingleGoalGridState(
                    neighbor_cell,
                    self.goal,
                    self.dist_from_start + 1,
                    self.use_heuristic,
                    self.maze
                )

                nbr_states.append(neighbor_state)
        # ------------------------------
        return nbr_states

    # TODO(V): fill in the is_goal, compute_heuristic, __hash__, and __eq__ methods
    # Your heuristic should be the manhattan distance between the state and the goal

    # Checks if goal has been reached
    def is_goal(self) -> bool:
        # In python "==" performs deep tuple equality checking, so this works as desired
        return self.state == self.goal

    def __hash__(self) -> int:
        return hash(self.state)
    def __eq__(self, other) -> bool:
        return self.state == other.state

    def compute_heuristic(self) -> float:
        return manhattan(self.state, self.goal)
    
    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state)
    def __repr__(self):
        return str(self.state)

# MultiGoalGridState --------------------------------------------------------------------------------

class MultiGoalGridState(AbstractState):
    # state: a length 2 tuple indicating the current location in the grid, e.g., (row, col)
    # goal: a tuple of length 2 tuples of locations in the grid that have not yet been reached
    #       e.g., ((row1, col1), (row2, col2), ...)
    # maze: Maze object for the problem. You can call maze.neighboring_cells(...) and maze.is_free(...)
    # mst_cache: reference to a dictionary which caches a set of goal locations to their MST value
    def __init__(self, state, goal, dist_from_start, use_heuristic, maze, mst_cache):
        self.maze = maze
        self.mst_cache = mst_cache
        super().__init__(state, goal, dist_from_start, use_heuristic)
        
    # Generate neighboring MultiGoalGridState objects
    def get_neighbors(self) -> list[MultiGoalGridState]:
        nbr_states = []
        neighboring_cells = self.maze.neighboring_cells(*self.state)
        # TODO(VI): fill this in
        # -------------------------------

        # -------------------------------
        return nbr_states

    # TODO(VI): fill in the is_goal, compute_heuristic, __hash__, and __eq__ methods
    
    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state) + ", goals=" + str(self.goal)
    def __repr__(self):
        return str(self.state) + ", goals=" + str(self.goal)
