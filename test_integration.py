# This script will test the integration of the NextGenJAX model with the problem generation module.
# It will attempt to generate a set of mathematical problems and check for errors.

import sys
sys.path.append('src')

import jax
import problem_generation

def test_problem_generation():
    # Initialize the NextGenJAX model
    model = problem_generation.initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)
    dummy_input = jax.numpy.zeros((1, 512))  # Changed from (1, 128) to (1, 512)
    params = model.init(rng_key, dummy_input)

    # Generate a set of problems
    problems = []
    for _ in range(5):
        rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
        algebra_problem, _ = problem_generation.generate_algebra_problem(params, subkey1)
        calculus_problem, _ = problem_generation.generate_calculus_problem(params, subkey2)
        problems.extend([algebra_problem, calculus_problem])

    # Check for any errors in the problems generated
    assert all(problem is not None for problem in problems), "Some problems were not generated correctly."

def test_simple_arithmetic():
    # Initialize the NextGenJAX model
    model = problem_generation.initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)
    dummy_input = jax.numpy.zeros((1, 512))  # Changed from (1, 128) to (1, 512)
    params = model.init(rng_key, dummy_input)

    # Test the expression "2+2"
    rng_key, subkey = jax.random.split(rng_key)
    input_array = jax.numpy.array([[2, 2] + [0] * 510])  # Padding with zeros to match (1, 512) shape
    result = model.apply(params, subkey, input_array, method=model.solve_algebra)

    # Check if the output is 4
    assert jax.numpy.allclose(result, jax.numpy.array([4])), f"Expected [4], but got {result}"

if __name__ == "__main__":
    test_problem_generation()
    test_simple_arithmetic()
    print("All tests passed successfully!")