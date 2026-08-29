from state import WordLadderState, EightPuzzleState, SingleGoalGridState, MultiGoalGridState
from search import best_first_search
from utils import read_puzzle, read_word_ladders, draw_eight_puzzle_path
from maze import Maze

import time
import argparse

def main(args):
    if args.problem_type == "WordLadder":
        word_ladder_problems = read_word_ladders()
        # each action costs 1
        cost_per_letter = {
            chr(c_idx): 1 for c_idx in range(97, 97+26)
        }
        # we have an option to add cost to vowels, making it better not to use vowels in the path
        if args.add_cost_to_vowels:
            for vowel in "aeiou":
                cost_per_letter[vowel] = 10
        for start_word, goal_word in word_ladder_problems:
            print(f"Doing WordLadder from {start_word} to {goal_word}")
            start = time.time()
            starting_state = WordLadderState(
                start_word, goal_word, 
                dist_from_start=0, use_heuristic=args.use_heuristic,
                cost_per_letter=cost_per_letter)
            path = best_first_search(starting_state)
            end = time.time()
            print("\tPath length: ", len(path))
            print("\tPath found:", [p for p in path])
            print(f"\tTime: {end-start:.3f}")

    elif args.problem_type == "EightPuzzle":
        print(f"Doing EightPuzzle for length {args.puzzle_len} puzzles")
        all_puzzles = read_puzzle(f"data/eight_puzzle/{args.puzzle_len}_moves.txt")
        for puzzle in all_puzzles:
            start = time.time()
            start_puzzle = puzzle[0]
            zero_loc = puzzle[1]
            print(f"Start puzzle: {start_puzzle}")
            goal_puzzle = [[0,1,2],[3,4,5],[6,7,8]]
            starting_state = EightPuzzleState(start_puzzle, goal_puzzle, 
                                dist_from_start=0, use_heuristic=args.use_heuristic, zero_loc=zero_loc)
            path = best_first_search(starting_state)
            end = time.time()
            print("\tPath length: ", len(path))
            print(f"\tTime: {end-start:.3f}")
            if args.draw_solution:
                draw_eight_puzzle_path(path)

    elif args.problem_type in ("GridSingle", "GridMultiGoal"):
        filename = args.maze_file
        print(f"Doing Maze search for file {filename}")
        maze = Maze(filename, allow_waiting=args.allow_waiting)
        # maze.starts is a dictionary mapping characters to locations
        # maze.goals is a dictionary mapping characters to a tuple of locations
        tasks = [(maze.starts[c], maze.goals[c]) for c in maze.starts]
        starts, goals = zip(*tasks)
        if args.draw_maze_only:
            maze.draw_maze(save=args.save_maze, show=(args.show_maze_vis or args.save_maze is None))
            return

        path = []
        start_time = time.time()  
        if args.problem_type == "GridSingle":
            # single goal search
            start = starts[0] # only one start
            goal = goals[0][0] # only one goal for that one start
            starting_state = SingleGoalGridState(
                start, goal, 
                dist_from_start = 0, use_heuristic = args.use_heuristic, 
                maze = maze)
        elif args.problem_type == "GridMultiGoal":
            # multi goal search
            start = starts[0] # only one start
            goal = goals[0] # only need goals for the one start
            starting_state = MultiGoalGridState(
                start, goal, 
                dist_from_start = 0, use_heuristic = args.use_heuristic,
                maze = maze, mst_cache = {})
        
        # regardless of problem type, we can use the same search function
        path = best_first_search(starting_state)
        end_time = time.time()

        print("\tStart: ", start)
        print("\tGoal: ", goal)
        print("\tPath length: ", len(path))
        print("\tStates explored: ", maze.num_states_validated)
        print("\tTime:", end_time-start_time)
        
        if args.show_maze_vis or args.save_maze:
            maze.draw_maze(path=path, save=args.save_maze, show=args.show_maze_vis)
    else:
        print("Problem type must be one of [WordLadder, EightPuzzle, GridSingle, GridMultiGoal]")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 Online MP1 Search', allow_abbrev=False)
    parser.add_argument('--h', action='help', help=argparse.SUPPRESS)
    parser.add_argument('--problem_type',dest="problem_type", type=str,default="WordLadder",
                        help='Which search problem (i.e., AbstractState) to solve: [WordLadder, EightPuzzle, GridSingle, GridMultiGoal]')
    parser.add_argument('--use_heuristic', action = 'store_true',
                        help = 'use heuristic h in best_first_search')
    parser.add_argument('--add_cost_to_vowels', action = 'store_true',
                        help = 'make vowels cost 10 instead of 1')
    parser.add_argument('--puzzle_len',dest="puzzle_len", type=int, default = 5,
                        help='EightPuzzle problem difficulty: one of [5, 10, 27]')
    parser.add_argument('--draw_solution', action = 'store_true',
                        help = 'draw the solution path for the EightPuzzle problem')
    parser.add_argument('--maze_file', type=str, default="data/grid_single/tiny",
                        help = 'path to maze file')
    parser.add_argument('--show_maze_vis', action = 'store_true',
                        help = 'show maze visualization')
    parser.add_argument('--draw_maze_only', action = 'store_true',
                        help = 'draw the maze without running search')
    parser.add_argument('--allow_waiting', action = 'store_true',
                        help = 'adds a waiting action to the agent')
    parser.add_argument('--save_maze', dest = 'save_maze', type = str, default = None,
                        help = 'save output to image file')

    args = parser.parse_args()
    print("Parsed Arguments:")
    for arg_name, arg_value in sorted(vars(args).items()):
        print(f"  {arg_name}: {arg_value}")
    main(args)
