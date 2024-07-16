# This script will test the integration of the problem generation module.
# It will attempt to generate a set of mathematical problems and check for errors.

import sys
sys.path.append('src')

import problem_generation

def test_simple_arithmetic():
    # Test the expression "2+2"
    problem, solution = problem_generation.generate_algebra_problem("2+2")

    # Check if the problem is correctly generated
    assert problem == "2+2", f"Expected problem '2+2', but got '{problem}'"

    # Check if the solution is correct
    assert solution == "4", f"Expected solution '4', but got '{solution}'"

    print("Simple arithmetic test passed successfully!")

def test_problem_generation():
    # Generate a set of problems
    problems = []
    solutions = []
    for _ in range(5):
        algebra_problem, algebra_solution = problem_generation.generate_algebra_problem()
        calculus_problem, calculus_solution = problem_generation.generate_calculus_problem()
        problems.extend([algebra_problem, calculus_problem])
        solutions.extend([algebra_solution, calculus_solution])

    # Check for any errors in the problems generated
    assert all(problem is not None for problem in problems), "Some problems were not generated correctly."
    assert all(solution is not None for solution in solutions), "Some solutions were not generated correctly."

    print("Problem generation test passed successfully!")

if __name__ == "__main__":
    test_simple_arithmetic()
    test_problem_generation()
    print("All tests passed successfully!")