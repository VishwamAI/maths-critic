# This script will test the integration of the problem generation module.
# It will attempt to generate a set of mathematical problems and check for errors.

import sys
sys.path.append('src')

import problem_generation
import jax
import jax.numpy

def test_simple_arithmetic():
    # Initialize the NextGenJAX model
    model = problem_generation.initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)
    dummy_input = jax.numpy.zeros((1, 512))
    params = model.init(rng_key, dummy_input)

    # Generate a new rng_key for problem generation
    rng_key, subkey = jax.random.split(rng_key)

    # Test the expression "2+2"
    problem, solution = problem_generation.generate_algebra_problem(params, subkey, "2+2")

    # Check if the problem is correctly generated
    assert problem == "2+2", f"Expected problem '2+2', but got '{problem}'"

    # Check if the solution is correct
    assert str(solution) == "4", f"Expected solution '4', but got '{solution}'"

    print("Simple arithmetic test passed successfully!")

def test_problem_generation():
    # Initialize the NextGenJAX model
    model = problem_generation.initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)
    dummy_input = jax.numpy.zeros((1, 512))
    params = model.init(rng_key, dummy_input)

    # Generate a set of problems
    problems = []
    solutions = []
    for _ in range(5):
        rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
        algebra_problem, algebra_solution = problem_generation.generate_algebra_problem(params, subkey1)
        calculus_problem, calculus_solution = problem_generation.generate_calculus_problem(params, subkey2, rng_key)
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