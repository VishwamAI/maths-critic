# This script will test the integration of the NextGenJAX model with the problem generation module.
# It will attempt to generate a set of mathematical problems and check for errors.

import sys
sys.path.append('src')

import problem_generation

def test_problem_generation():
    # Initialize the NextGenJAX model
    problem_generation.initialize_nextgenjax_model()

    # Generate a set of problems
    problems = [problem_generation.generate_problem() for _ in range(10)]

    # Check for any errors in the problems generated
    assert all(problem is not None for problem in problems), "Some problems were not generated correctly."

def test_simple_arithmetic():
    # Initialize the NextGenJAX model
    problem_generation.initialize_nextgenjax_model()

    # Test the expression "2+2"
    result = problem_generation.solve_problem("2+2")

    # Check if the output is "4"
    assert result == "4", f"Expected '4', but got '{result}'"

if __name__ == "__main__":
    test_problem_generation()
    test_simple_arithmetic()
    print("All tests passed successfully!")