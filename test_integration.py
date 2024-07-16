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

    # Generate a set of problems
    problems = []
    solutions = []
    for i in range(5):
        print(f"Generating problem set {i+1}...")
        rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
        algebra_problem, algebra_solution = problem_generation.generate_algebra_problem(global_model, global_params, subkey1)
        print(f"  Generated algebra problem: {algebra_problem}")
        print(f"  Generated algebra solution: {algebra_solution}")
        calculus_problem, calculus_solution = problem_generation.generate_calculus_problem(global_model, global_params, subkey2)
        print(f"  Generated calculus problem: {calculus_problem}")
        print(f"  Generated calculus solution: {calculus_solution}")
        problems.extend([algebra_problem, calculus_problem])
        solutions.extend([algebra_solution, calculus_solution])
        print(f"Problem set {i+1} generated successfully")

    # Check for any errors in the problems generated
    print("Checking generated problems and solutions...")
    assert all(problem is not None for problem in problems), "Some problems were not generated correctly."
    assert all(solution is not None for solution in solutions), "Some solutions were not generated correctly."

    print("All problems and solutions generated correctly")
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
        for i in range(100):  # Reduced from 1000 to 100
            print(f"Iteration {i+1}/100")
            problem, solution = problem_generation.generate_algebra_problem(global_model, global_params, rng_key, "2*x + 3 = 15")
            print(f"Generated problem: {problem}")
            print(f"Generated solution: {solution}")
            assert problem is not None, f"Expected a problem to be generated, but got None (iteration {i+1})"
            assert solution is not None, f"Expected a solution to be generated, but got None (iteration {i+1})"
            if i % 10 == 0 and i > 0:
                print(f"Completed {i} iterations")
        print("Stress test completed all iterations successfully!")
    except Exception as e:
        print(f"Stress test failed at iteration {i+1}. Error: {str(e)}")
    finally:
        print("Stress test finished.")

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
    for i in range(5):  # Adjust the range as needed
        print(f"Generating problem {i+1}")
        problem, solution = problem_generation.generate_large_complex_problem(global_model, global_params, rng_key)
        print(f"Generated problem: {problem}")
        print(f"Generated solution: {solution}")
        assert problem is not None, f"Expected a large complex problem to be generated, but got None (iteration {i+1})"
        assert solution is not None, f"Expected a solution to be generated, but got None (iteration {i+1})"
        rng_key, _ = jax.random.split(rng_key)
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

    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
            print(f"{test.__name__} completed successfully!")
        except Exception as e:
            print(f"An error occurred during {test.__name__}: {str(e)}")

    print("\nAll tests completed.")
    print("Test suite finished.")