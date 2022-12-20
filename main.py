import os
import glob
import argparse
import time 

from VRPTW import *


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Solving VRPTW with algorithms')
    parser.add_argument('problem_file', type=str, help='Problem file (in Solomon format)')
    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    if not os.path.exists('solutions'):
        os.mkdir('solutions')
    assert os.path.exists(args.problem_file), "Problem file doesn't exist"
    problem = SolomonFormatParser(args.problem_file).get_problem()
    print(problem)
    solution = IteratedLocalSearch(problem).execute()
    with open(f"""solutions/{args.problem_file.split(os.sep)[-1].split(".")[0]}-ILS.sol""", 'w') as f:
        f.write(problem.print_canonical(solution))

    print('\nCalculating optimal route using Genetic Algorithm .... \n')
    print(problem)
    problem = SolomonFormatParser(args.problem_file).get_problem()
    solution = GASearch(problem).execute(args.problem_file)
