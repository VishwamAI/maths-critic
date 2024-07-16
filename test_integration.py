# This script will test the integration of the problem generation module.
# It will attempt to generate a set of mathematical problems and check for errors.

import sys
sys.path.append('src')

import problem_generation
import jax
import jax.numpy
import numpy as np
from scipy import stats

def initialize_nextgenjax_model():
    model = problem_generation.initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)
    dummy_input = jax.numpy.zeros((1, 512))
    params = model.init(rng_key, dummy_input)
    return model, params

def test_simple_arithmetic():
    model, params = initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)

    problem, expected_solution = "2+2", "4"
    generated_problem, generated_solution = problem_generation.generate_algebra_problem(params, rng_key, problem)

    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"
    assert str(generated_solution) == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"

    print("Simple arithmetic test passed successfully!")

def test_complex_arithmetic():
    model, params = initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)

    problem, expected_solution = "3 * (4 + 2) / 2", "9"
    generated_problem, generated_solution = problem_generation.generate_complex_arithmetic_problem(model, params, rng_key)

    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"
    assert str(generated_solution) == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"

    print("Complex arithmetic test passed successfully!")

def test_basic_algebra():
    model, params = initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)

    problem, expected_solution = "2x + 3 = 11", "x = 4"
    generated_problem, generated_solution = problem_generation.generate_basic_algebra_problem(model, params, rng_key)

    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"
    assert generated_solution == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"

    print("Basic algebra test passed successfully!")

def test_quadratic_equations():
    model, params = initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)

    problem, expected_solution = "x^2 + 5x + 6 = 0", "x = -2 or x = -3"
    generated_problem, generated_solution = problem_generation.generate_quadratic_equation_problem(model, params, rng_key)

    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"
    assert generated_solution == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"

    print("Quadratic equations test passed successfully!")

def test_basic_differentiation():
    model, params = initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)

    problem, expected_solution = "d/dx (x^2 + 3x)", "2x + 3"
    generated_problem, generated_solution = problem_generation.generate_basic_differentiation_problem(model, params, rng_key)

    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"
    assert generated_solution == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"

    print("Basic differentiation test passed successfully!")

def test_basic_integration():
    model, params = initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)

    problem, expected_solution = "∫ (2x + 1) dx", "x^2 + x + C"
    generated_problem, generated_solution = problem_generation.generate_basic_integration_problem(model, params, rng_key)

    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"
    assert generated_solution == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"

    print("Basic integration test passed successfully!")

def test_problem_generation():
    model, params = initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)

    # Generate a set of problems
    problems = []
    solutions = []
    for _ in range(5):
        rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
        algebra_problem, algebra_solution = problem_generation.generate_algebra_problem(params, subkey1)
        calculus_problem, calculus_solution = problem_generation.generate_calculus_problem(model, params, subkey2)
        problems.extend([algebra_problem, calculus_problem])
        solutions.extend([algebra_solution, calculus_solution])

    # Check for any errors in the problems generated
    assert all(problem is not None for problem in problems), "Some problems were not generated correctly."
    assert all(solution is not None for solution in solutions), "Some solutions were not generated correctly."

    print("Problem generation test passed successfully!")

def test_problem_generation_edge_cases():
    model, params = initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)

    # Test with empty input
    empty_problem, empty_solution = problem_generation.generate_algebra_problem(model, params, rng_key, "")
    assert empty_problem is None, f"Expected no problem to be generated for empty input, but got: {empty_problem}"
    assert empty_solution is None, f"Expected no solution to be generated for empty input, but got: {empty_solution}"

    # Test with invalid input
    invalid_problem, invalid_solution = problem_generation.generate_algebra_problem(model, params, rng_key, "invalid input")
    assert invalid_problem is None, f"Expected no problem to be generated for invalid input, but got: {invalid_problem}"
    assert invalid_solution is None, f"Expected no solution to be generated for invalid input, but got: {invalid_solution}"

    print("Edge case tests passed successfully!")

# Add stress test function for problem generation
def test_problem_generation_stress():
    model, params = initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)

    # Generate a large number of problems to test stability and performance
    for _ in range(1000):
        problem, solution = problem_generation.generate_algebra_problem(model, params, rng_key, "2*x + 3 = 15")
        assert problem is not None, "Expected a problem to be generated, but got None"
        assert solution is not None, "Expected a solution to be generated, but got None"

    print("Stress test passed successfully!")

# Add new test function for advanced mathematical domains
def test_advanced_math_problems():
    model, params = initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)

    # Test advanced algebra problem
    advanced_algebra_problem, advanced_algebra_solution = problem_generation.generate_advanced_algebra_problem(model, params, rng_key, "solve for x: x^2 + 4*x - 5 = 0")
    assert advanced_algebra_problem is not None, "Expected an advanced algebra problem to be generated, but got None"
    assert advanced_algebra_solution is not None, "Expected an advanced algebra solution to be generated, but got None"

    # Test advanced calculus problem
    advanced_calculus_problem, advanced_calculus_solution = problem_generation.generate_advanced_calculus_problem(model, params, rng_key, "integrate: sin(x) dx from 0 to pi")
    assert advanced_calculus_problem is not None, "Expected an advanced calculus problem to be generated, but got None"
    assert advanced_calculus_solution is not None, "Expected an advanced calculus solution to be generated, but got None"

    print("Advanced math problems test passed successfully!")

def test_higher_order_differential_equations():
    model, params = initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)
    problem, solution = problem_generation.generate_higher_order_differential_equation(model, params, rng_key)
    assert problem is not None, "Expected a higher-order differential equation problem to be generated, but got None"
    assert solution is not None, "Expected a solution to be generated, but got None"
    print("Higher-order differential equations test passed successfully!")

def test_multivariable_calculus():
    model, params = initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)
    problem, solution = problem_generation.generate_multivariable_calculus_problem(model, params, rng_key)
    assert problem is not None, "Expected a multivariable calculus problem to be generated, but got None"
    assert solution is not None, "Expected a solution to be generated, but got None"
    print("Multivariable calculus test passed successfully!")

def test_advanced_statistics():
    model, params = initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)
    problem, solution = problem_generation.generate_advanced_statistics_problem(model, params, rng_key)
    assert problem is not None, "Expected an advanced statistics problem to be generated, but got None"
    assert solution is not None, "Expected a solution to be generated, but got None"
    print("Advanced statistics test passed successfully!")

def test_large_complex_problems():
    model, params = initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(0)
    problem, solution = problem_generation.generate_large_complex_problem(model, params, rng_key)
    assert problem is not None, "Expected a large complex problem to be generated, but got None"
    assert solution is not None, "Expected a solution to be generated, but got None"
    print("Large complex problems test passed successfully!")

if __name__ == "__main__":
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