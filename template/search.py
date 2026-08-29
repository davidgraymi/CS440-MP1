import heapq
from state import AbstractState
# You do not need any other imports

def best_first_search(starting_state : AbstractState) -> list[AbstractState]:
    '''
    Implementation of best first search algorithm

    Input:
        starting_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''
    # we will use this visited_states dictionary to serve multiple purposes
    # - visited_states[state] = (parent_state, distance_of_state_from_start)
    #   - keep track of which states have been visited by the search algorithm
    #   - keep track of the parent of each state, so we can call backtrack(visited_states, goal_state) and obtain the path
    #   - keep track of the distance of each state from start node
    #       - if we find a shorter path to the same state we can update with the new state 
    # NOTE: we can hash states because the __hash__/__eq__ method of AbstractState is implemented
    visited_states = {starting_state: (None, 0)}

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = []
    heapq.heappush(frontier, starting_state)
    
    # TODO(III): implement the rest of the best first search algorithm
    # Your code here ---------------

    # ------------------------------
    
    # if you do not find the goal return an empty list
    return []

def backtrack(visited_states: dict, goal_state: AbstractState) -> list[AbstractState]:
    '''
    Implementation of the backtrack method

    Input:
        visited_states: a dictionary mapping AbstractState objects to (parent_state, distance_from_start) tuples
        goal_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''
    path = []
    # TODO(III): implement the backtrack method using the parent pointers in visited_states
    # Your code here ---------------

    # ------------------------------
    return path
