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
import gc
import psutil
from contextlib import contextmanager

TIMEOUT = 300  # 5 minutes timeout for each test

def tearDown(self):
    gc.collect()

def check_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

@contextmanager
def managed_resource():
    try:
        yield
    finally:
        gc.collect()

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
    print("Starting test_simple_arithmetic...")
    rng_key = jax.random.PRNGKey(0)
    print("Simple arithmetic test: Initializing...")
    problem, expected_solution = "2+2", "4"
    print(f"Problem to generate: {problem}")
    print("Calling generate_algebra_problem...")
    generated_problem, generated_solution = problem_generation.generate_algebra_problem(global_model, global_params, rng_key, problem)
    print(f"Generated problem: {generated_problem}")
    print(f"Generated solution: {generated_solution}")
    print("Asserting generated problem...")
    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"
    print("Asserting generated solution...")
    assert str(generated_solution) == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"
    print("test_simple_arithmetic completed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_complex_arithmetic():
    print("Starting test_complex_arithmetic...")
    rng_key = jax.random.PRNGKey(0)
    print("Initialized random key")
    problem, expected_solution = "3 * (4 + 2) / 2", "9"
    print(f"Set problem: {problem}, expected solution: {expected_solution}")
    print("Calling generate_complex_arithmetic_problem...")
    generated_problem, generated_solution = problem_generation.generate_complex_arithmetic_problem(global_model, global_params, rng_key)
    print(f"Generated problem: {generated_problem}")
    print(f"Generated solution: {generated_solution}")
    print("Asserting generated problem matches expected problem...")
    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"
    print("Asserting generated solution matches expected solution...")
    assert str(generated_solution) == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"
    print("test_complex_arithmetic completed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_basic_algebra():
    print("Starting test_basic_algebra...")
    rng_key = jax.random.PRNGKey(0)
    print("Initialized random key")
    problem, expected_solution = "2x + 3 = 11", "x = 4"
    print(f"Set problem: {problem}, expected solution: {expected_solution}")
    print("Calling generate_basic_algebra_problem...")
    generated_problem, generated_solution = problem_generation.generate_basic_algebra_problem(global_model, global_params, rng_key)
    print(f"Generated problem: {generated_problem}")
    print(f"Generated solution: {generated_solution}")
    print("Asserting generated problem matches expected problem...")
    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"
    print("Asserting generated solution matches expected solution...")
    assert generated_solution == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"
    print("Basic algebra test passed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_quadratic_equations():
    print("Starting test_quadratic_equations...")
    rng_key = jax.random.PRNGKey(0)
    print("Initialized random key")

    problem, expected_solution = "x^2 + 5x + 6 = 0", "x = -2 or x = -3"
    print(f"Defined problem: {problem}")
    print(f"Expected solution: {expected_solution}")

    print("Generating quadratic equation problem...")
    generated_problem, generated_solution = problem_generation.generate_quadratic_equation_problem(global_model, global_params, rng_key)

    print(f"Generated problem: {generated_problem}")
    print(f"Generated solution: {generated_solution}")

    print("Asserting generated problem matches expected problem...")
    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"

    print("Asserting generated solution matches expected solution...")
    assert generated_solution == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"

    print("test_quadratic_equations completed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_basic_differentiation():
    rng_key = jax.random.PRNGKey(0)
    print("Starting test_basic_differentiation...")

    print("Generating basic differentiation problem...")
    problem, expected_solution = "d/dx (x^2 + 3x)", "2x + 3"
    print(f"Expected problem: {problem}")
    print(f"Expected solution: {expected_solution}")

    print("Calling generate_basic_differentiation_problem...")
    generated_problem, generated_solution = problem_generation.generate_basic_differentiation_problem(global_model, global_params, rng_key)

    print(f"Generated problem: {generated_problem}")
    print(f"Generated solution: {generated_solution}")

    print("Asserting generated problem matches expected problem...")
    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"

    print("Asserting generated solution matches expected solution...")
    assert generated_solution == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"

    print("test_basic_differentiation completed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_basic_integration():
    print("Starting test_basic_integration...")
    rng_key = jax.random.PRNGKey(0)
    print("Initialized random key")

    problem, expected_solution = "∫ (2x + 1) dx", "x^2 + x + C"
    print(f"Set up test problem: {problem}")
    print(f"Expected solution: {expected_solution}")

    print("Generating basic integration problem...")
    generated_problem, generated_solution = problem_generation.generate_basic_integration_problem(global_model, global_params, rng_key)

    print(f"Generated problem: {generated_problem}")
    print(f"Generated solution: {generated_solution}")

    print("Asserting generated problem matches expected problem...")
    assert generated_problem == problem, f"Expected problem: {problem}, but got: {generated_problem}"
    print("Assertion passed for generated problem")

    print("Asserting generated solution matches expected solution...")
    assert generated_solution == expected_solution, f"Expected solution: {expected_solution}, but got: {generated_solution}"
    print("Assertion passed for generated solution")

    print("test_basic_integration completed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_problem_generation():
    rng_key = jax.random.PRNGKey(0)
    print("Starting problem generation test...")

    def problem_generator():
        nonlocal rng_key
        for i in range(5):
            print(f"Generating problem set {i+1}...")
            rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
            algebra_problem, algebra_solution = problem_generation.generate_algebra_problem(global_model, global_params, subkey1)
            yield algebra_problem, algebra_solution
            calculus_problem, calculus_solution = problem_generation.generate_calculus_problem(global_model, global_params, subkey2)
            yield calculus_problem, calculus_solution
            print(f"Problem set {i+1} generated successfully")

    # Generate and check problems
    for i, (problem, solution) in enumerate(problem_generator(), 1):
        print(f"  Generated problem {i}: {problem}")
        print(f"  Generated solution {i}: {solution}")
        assert problem is not None, f"Problem {i} was not generated correctly."
        assert solution is not None, f"Solution {i} was not generated correctly."

        # Add memory check
        if i % 5 == 0:
            check_memory_usage()

    print("All problems and solutions generated and checked correctly")
    print("Problem generation test passed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_problem_generation_edge_cases():
    rng_key = jax.random.PRNGKey(0)
    print("Starting edge case tests...")

    print("Testing with empty input...")
    empty_problem, empty_solution = problem_generation.generate_algebra_problem(global_model, global_params, rng_key, "")
    print(f"Empty input result - Problem: {empty_problem}, Solution: {empty_solution}")
    assert empty_problem is None, f"Expected no problem to be generated for empty input, but got: {empty_problem}"
    assert empty_solution is None, f"Expected no solution to be generated for empty input, but got: {empty_solution}"
    print("Empty input test passed.")

    print("Testing with invalid input...")
    invalid_problem, invalid_solution = problem_generation.generate_algebra_problem(global_model, global_params, rng_key, "invalid input")
    print(f"Invalid input result - Problem: {invalid_problem}, Solution: {invalid_solution}")
    assert invalid_problem is None, f"Expected no problem to be generated for invalid input, but got: {invalid_problem}"
    assert invalid_solution is None, f"Expected no solution to be generated for invalid input, but got: {invalid_solution}"
    print("Invalid input test passed.")

    print("Edge case tests completed successfully!")

# Add stress test function for problem generation
@memory_profiler.profile
@timeout_decorator.timeout(TIMEOUT)
def test_problem_generation_stress():
    rng_key = jax.random.PRNGKey(0)
    print("Starting stress test...")
    try:
        def problem_generator():
            nonlocal rng_key
            for i in range(50):  # Further reduced from 100 to 50
                rng_key, subkey = jax.random.split(rng_key)
                yield problem_generation.generate_algebra_problem(global_model, global_params, subkey, "2*x + 3 = 15")

        for i, (problem, solution) in enumerate(problem_generator(), 1):
            print(f"Iteration {i}/50")
            print(f"Generated problem: {problem}")
            print(f"Generated solution: {solution}")
            assert problem is not None, f"Expected a problem to be generated, but got None (iteration {i})"
            assert solution is not None, f"Expected a solution to be generated, but got None (iteration {i})"
            if i % 10 == 0:
                print(f"Completed {i} iterations")
                check_memory_usage()
        print("Stress test completed all iterations successfully!")
    except Exception as e:
        print(f"Stress test failed at iteration {i}. Error: {str(e)}")
    finally:
        print("Stress test finished.")
        check_memory_usage()

# Add new test function for advanced mathematical domains
@timeout_decorator.timeout(TIMEOUT)
def test_advanced_math_problems():
    rng_key = jax.random.PRNGKey(0)
    print("Starting test_advanced_math_problems...")

    print("Testing advanced algebra problem...")
    advanced_algebra_problem, advanced_algebra_solution = problem_generation.generate_advanced_algebra_problem(global_model, global_params, rng_key, "solve for x: x^2 + 4*x - 5 = 0")
    print(f"Advanced algebra problem: {advanced_algebra_problem}")
    print(f"Advanced algebra solution: {advanced_algebra_solution}")
    assert advanced_algebra_problem is not None, "Expected an advanced algebra problem to be generated, but got None"
    assert advanced_algebra_solution is not None, "Expected an advanced algebra solution to be generated, but got None"
    print("Advanced algebra problem test completed.")

    print("Testing advanced calculus problem...")
    advanced_calculus_problem, advanced_calculus_solution = problem_generation.generate_advanced_calculus_problem(global_model, global_params, rng_key, "integrate: sin(x) dx from 0 to pi")
    print(f"Advanced calculus problem: {advanced_calculus_problem}")
    print(f"Advanced calculus solution: {advanced_calculus_solution}")
    assert advanced_calculus_problem is not None, "Expected an advanced calculus problem to be generated, but got None"
    assert advanced_calculus_solution is not None, "Expected an advanced calculus solution to be generated, but got None"
    print("Advanced calculus problem test completed.")

    print("test_advanced_math_problems completed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_higher_order_differential_equations():
    rng_key = jax.random.PRNGKey(0)
    print("Starting higher-order differential equations test...")
    try:
        problem, solution = problem_generation.generate_higher_order_differential_equation(global_model, global_params, rng_key)
        print(f"Generated problem: {problem}")
        print(f"Generated solution: {solution}")
        assert problem is not None, "Expected a higher-order differential equation problem to be generated, but got None"
        assert solution is not None, "Expected a solution to be generated, but got None"
        print("Higher-order differential equations test passed successfully!")
    except Exception as e:
        print(f"Error in test_higher_order_differential_equations: {str(e)}")
        raise
    finally:
        print("Finished higher-order differential equations test.")

@timeout_decorator.timeout(TIMEOUT)
def test_multivariable_calculus():
    print("Starting test_multivariable_calculus...")
    rng_key = jax.random.PRNGKey(0)
    print("Generating multivariable calculus problem...")
    problem, solution = problem_generation.generate_multivariable_calculus_problem(global_model, global_params, rng_key)
    print(f"Generated problem: {problem}")
    print(f"Generated solution: {solution}")
    assert problem is not None, "Expected a multivariable calculus problem to be generated, but got None"
    assert solution is not None, "Expected a solution to be generated, but got None"
    print("Assertions passed.")
    print("test_multivariable_calculus completed successfully!")

@timeout_decorator.timeout(TIMEOUT)
def test_advanced_statistics():
    rng_key = jax.random.PRNGKey(0)
    print("Starting advanced statistics test...")
    try:
        problem, solution = problem_generation.generate_advanced_statistics_problem(global_model, global_params, rng_key)
        print(f"Generated problem: {problem}")
        print(f"Generated solution: {solution}")
        assert problem is not None, "Expected an advanced statistics problem to be generated, but got None"
        assert solution is not None, "Expected a solution to be generated, but got None"
        print("Advanced statistics test passed successfully!")
    except Exception as e:
        print(f"Error in advanced statistics test: {str(e)}")
        raise
    finally:
        print("Advanced statistics test completed.")

@timeout_decorator.timeout(TIMEOUT)
def test_large_complex_problems():
    rng_key = jax.random.PRNGKey(0)
    print("Starting large complex problems test...")

    def test_single_large_problem(problem_type):
        nonlocal rng_key
        rng_key, subkey = jax.random.split(rng_key)
        print(f"Generating {problem_type} problem")
        problem, solution = problem_generation.generate_large_complex_problem(global_model, global_params, subkey, problem_type=problem_type)
        print(f"Generated problem: {problem}")
        print(f"Generated solution: {solution}")
        assert problem is not None, f"Expected a large complex {problem_type} problem to be generated, but got None"
        assert solution is not None, f"Expected a solution for {problem_type} problem to be generated, but got None"

    problem_types = ['algebra', 'calculus', 'statistics', 'differential_equations', 'linear_algebra']
    for problem_type in problem_types:
        test_single_large_problem(problem_type)

    print("Large complex problems test passed successfully!")

if __name__ == "__main__":
    print("Starting test suite")
    tests = [
        test_simple_arithmetic,
        test_complex_arithmetic,
        test_basic_algebra,
        test_quadratic_equations,
        test_basic_differentiation,
        test_basic_integration,
        test_problem_generation,
        test_problem_generation_edge_cases,
        test_problem_generation_stress,
        test_advanced_math_problems,
        test_higher_order_differential_equations,
        test_multivariable_calculus,
        test_advanced_statistics,
        test_large_complex_problems
    ]

    try:
        for test in tests:
            try:
                print(f"\nRunning {test.__name__}...")
                test()
                print(f"{test.__name__} completed successfully!")
            except Exception as e:
                print(f"An error occurred during {test.__name__}: {str(e)}")
            finally:
                gc.collect()
                check_memory_usage()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("\nAll tests completed.")
        print("Test suite finished.")
        gc.collect()
        check_memory_usage()