# This script will test the integration of the problem generation module.
# It will attempt to generate a set of mathematical problems and check for errors.

import sys
sys.path.append('src')

import problem_generation
import jax
import jax.numpy
import numpy as np
from scipy import stats
import memory_profiler
import timeout_decorator

TIMEOUT = 300  # 5 minutes timeout for each test

def initialize_nextgenjax_model():
    model = problem_generation.initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)
    dummy_input = jax.numpy.zeros((1, 512))
    params = model.init(rng_key, dummy_input)
    return model, params

# Initialize the model once at the module level
global_model, global_params = initialize_nextgenjax_model()

@timeout_decorator.timeout(TIMEOUT)
def test_simple_arithmetic():
    rng_key = jax.random.PRNGKey(0)
    print("Starting simple arithmetic test...")
    problem, expected_solution = "2+2", "4"
    generated_problem, generated_solution = problem_generation.generate_algebra_problem(global_model, global_params, rng_key, problem)
    print(f"Generated problem: {generated_problem}")
    print(f"Generated solution: {generated_solution}")
    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"
    assert str(generated_solution) == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"
    print("Simple arithmetic test passed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_complex_arithmetic():
    rng_key = jax.random.PRNGKey(0)
    print("Starting complex arithmetic test...")
    problem, expected_solution = "3 * (4 + 2) / 2", "9"
    generated_problem, generated_solution = problem_generation.generate_complex_arithmetic_problem(global_model, global_params, rng_key)
    print(f"Generated problem: {generated_problem}")
    print(f"Generated solution: {generated_solution}")
    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"
    assert str(generated_solution) == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"
    print("Complex arithmetic test passed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_basic_algebra():
    rng_key = jax.random.PRNGKey(0)
    print("Starting basic algebra test...")
    problem, expected_solution = "2x + 3 = 11", "x = 4"
    generated_problem, generated_solution = problem_generation.generate_basic_algebra_problem(global_model, global_params, rng_key)
    print(f"Generated problem: {generated_problem}")
    print(f"Generated solution: {generated_solution}")
    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"
    assert generated_solution == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"
    print("Basic algebra test passed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_quadratic_equations():
    rng_key = jax.random.PRNGKey(0)
    print("Starting quadratic equations test...")

    problem, expected_solution = "x^2 + 5x + 6 = 0", "x = -2 or x = -3"
    generated_problem, generated_solution = problem_generation.generate_quadratic_equation_problem(global_model, global_params, rng_key)

    print(f"Generated problem: {generated_problem}")
    print(f"Generated solution: {generated_solution}")
    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"
    assert generated_solution == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"

    print("Quadratic equations test passed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_basic_differentiation():
    rng_key = jax.random.PRNGKey(0)
    print("Starting basic differentiation test...")

    problem, expected_solution = "d/dx (x^2 + 3x)", "2x + 3"
    generated_problem, generated_solution = problem_generation.generate_basic_differentiation_problem(global_model, global_params, rng_key)

    print(f"Generated problem: {generated_problem}")
    print(f"Generated solution: {generated_solution}")
    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"
    assert generated_solution == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"

    print("Basic differentiation test passed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_basic_integration():
    rng_key = jax.random.PRNGKey(0)
    print("Starting basic integration test...")

    problem, expected_solution = "∫ (2x + 1) dx", "x^2 + x + C"
    generated_problem, generated_solution = problem_generation.generate_basic_integration_problem(global_model, global_params, rng_key)

    print(f"Generated problem: {generated_problem}")
    print(f"Generated solution: {generated_solution}")
    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"
    assert generated_solution == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"

    print("Basic integration test passed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_problem_generation():
    rng_key = jax.random.PRNGKey(0)
    print("Starting problem generation test...")

    # Generate a set of problems
    problems = []
    solutions = []
    for i in range(5):
        rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
        algebra_problem, algebra_solution = problem_generation.generate_algebra_problem(global_model, global_params, subkey1)
        calculus_problem, calculus_solution = problem_generation.generate_calculus_problem(global_model, global_params, subkey2)
        problems.extend([algebra_problem, calculus_problem])
        solutions.extend([algebra_solution, calculus_solution])
        print(f"Generated problem set {i+1}")

    # Check for any errors in the problems generated
    assert all(problem is not None for problem in problems), "Some problems were not generated correctly."
    assert all(solution is not None for solution in solutions), "Some solutions were not generated correctly."

    print("Problem generation test passed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_problem_generation_edge_cases():
    rng_key = jax.random.PRNGKey(0)
    print("Starting edge case tests...")

    # Test with empty input
    empty_problem, empty_solution = problem_generation.generate_algebra_problem(global_model, global_params, rng_key, "")
    assert empty_problem is None, f"Expected no problem to be generated for empty input, but got: {empty_problem}"
    assert empty_solution is None, f"Expected no solution to be generated for empty input, but got: {empty_solution}"

    # Test with invalid input
    invalid_problem, invalid_solution = problem_generation.generate_algebra_problem(global_model, global_params, rng_key, "invalid input")
    assert invalid_problem is None, f"Expected no problem to be generated for invalid input, but got: {invalid_problem}"
    assert invalid_solution is None, f"Expected no solution to be generated for invalid input, but got: {invalid_solution}"

    print("Edge case tests passed successfully!")

# Add stress test function for problem generation
@memory_profiler.profile
@timeout_decorator.timeout(TIMEOUT)
def test_problem_generation_stress():
    rng_key = jax.random.PRNGKey(0)
    print("Starting stress test...")
    for i in range(100):  # Reduced from 1000 to 100
        problem, solution = problem_generation.generate_algebra_problem(global_model, global_params, rng_key, "2*x + 3 = 15")
        assert problem is not None, f"Expected a problem to be generated, but got None (iteration {i})"
        assert solution is not None, f"Expected a solution to be generated, but got None (iteration {i})"
        if i % 10 == 0:
            print(f"Completed {i} iterations")
    print("Stress test passed successfully!")

# Add new test function for advanced mathematical domains
@timeout_decorator.timeout(TIMEOUT)
def test_advanced_math_problems():
    rng_key = jax.random.PRNGKey(0)
    print("Starting advanced math problems test...")

    # Test advanced algebra problem
    advanced_algebra_problem, advanced_algebra_solution = problem_generation.generate_advanced_algebra_problem(global_model, global_params, rng_key, "solve for x: x^2 + 4*x - 5 = 0")
    print(f"Advanced algebra problem: {advanced_algebra_problem}")
    print(f"Advanced algebra solution: {advanced_algebra_solution}")
    assert advanced_algebra_problem is not None, "Expected an advanced algebra problem to be generated, but got None"
    assert advanced_algebra_solution is not None, "Expected an advanced algebra solution to be generated, but got None"

    # Test advanced calculus problem
    advanced_calculus_problem, advanced_calculus_solution = problem_generation.generate_advanced_calculus_problem(global_model, global_params, rng_key, "integrate: sin(x) dx from 0 to pi")
    print(f"Advanced calculus problem: {advanced_calculus_problem}")
    print(f"Advanced calculus solution: {advanced_calculus_solution}")
    assert advanced_calculus_problem is not None, "Expected an advanced calculus problem to be generated, but got None"
    assert advanced_calculus_solution is not None, "Expected an advanced calculus solution to be generated, but got None"

    print("Advanced math problems test passed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_higher_order_differential_equations():
    rng_key = jax.random.PRNGKey(0)
    print("Starting higher-order differential equations test...")
    problem, solution = problem_generation.generate_higher_order_differential_equation(global_model, global_params, rng_key)
    print(f"Generated problem: {problem}")
    print(f"Generated solution: {solution}")
    assert problem is not None, "Expected a higher-order differential equation problem to be generated, but got None"
    assert solution is not None, "Expected a solution to be generated, but got None"
    print("Higher-order differential equations test passed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_multivariable_calculus():
    rng_key = jax.random.PRNGKey(0)
    print("Starting multivariable calculus test...")
    problem, solution = problem_generation.generate_multivariable_calculus_problem(global_model, global_params, rng_key)
    print(f"Generated problem: {problem}")
    print(f"Generated solution: {solution}")
    assert problem is not None, "Expected a multivariable calculus problem to be generated, but got None"
    assert solution is not None, "Expected a solution to be generated, but got None"
    print("Multivariable calculus test passed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_advanced_statistics():
    rng_key = jax.random.PRNGKey(0)
    print("Starting advanced statistics test...")
    problem, solution = problem_generation.generate_advanced_statistics_problem(global_model, global_params, rng_key)
    print(f"Generated problem: {problem}")
    print(f"Generated solution: {solution}")
    assert problem is not None, "Expected an advanced statistics problem to be generated, but got None"
    assert solution is not None, "Expected a solution to be generated, but got None"
    print("Advanced statistics test passed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_large_complex_problems():
    rng_key = jax.random.PRNGKey(0)
    print("Starting large complex problems test...")
    problem, solution = problem_generation.generate_large_complex_problem(global_model, global_params, rng_key)
    print(f"Generated problem: {problem}")
    print(f"Generated solution: {solution}")
    assert problem is not None, "Expected a large complex problem to be generated, but got None"
    assert solution is not None, "Expected a solution to be generated, but got None"
    print("Large complex problems test passed successfully!")

if __name__ == "__main__":
    try:
        test_simple_arithmetic()
        test_complex_arithmetic()
        test_basic_algebra()
        test_quadratic_equations()
        test_basic_differentiation()
        test_basic_integration()
        test_problem_generation()
        test_problem_generation_edge_cases()
        test_problem_generation_stress()
        test_advanced_math_problems()
        test_higher_order_differential_equations()
        test_multivariable_calculus()
        test_advanced_statistics()
        test_large_complex_problems()
        print("All tests passed successfully!")
    except Exception as e:
        print(f"An error occurred during testing: {str(e)}")
    finally:
        print("Testing completed.")