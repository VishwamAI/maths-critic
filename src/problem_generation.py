import random
import haiku as hk
from nextgenjax.model import NextGenModel

def initialize_nextgenjax_model():
    """
    Initialize and return the NextGenModel with predefined hyperparameters using hk.transform.
    """
    def _model_fn():
        return NextGenModel(num_layers=6, hidden_size=512, num_heads=8, dropout_rate=0.1)

    return hk.transform(_model_fn)

model = initialize_nextgenjax_model()

def generate_algebra_problem(params, rng_key):
    """
    Generates a simple algebraic problem of the form ax + b = c using NextGenModel.
    Returns the problem as a string and the solution.
    """
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    c = random.randint(1, 10)

    problem = f"{a}x + {b} = {c}"
    solution = model.apply(params, rng_key, problem, method=model.solve_algebra)

    return problem, solution

def generate_calculus_problem(params, rng_key):
    """
    Generates a simple calculus problem of the form f(x) = integral(g(x)) using NextGenModel.
    Returns the problem as a string and the solution.
    """
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    c = random.randint(1, 10)

    g = f"{a}x^2 + {b}x + {c}"
    problem = f"f(x) = integral({g})"
    solution = model.apply(params, rng_key, problem, method=model.solve_calculus)

    return problem, solution

if __name__ == "__main__":
    import jax

    rng_key = jax.random.PRNGKey(0)
    dummy_input = "dummy input for initialization"
    params = model.init(rng_key, dummy_input)

    rng_key, subkey = jax.random.split(rng_key)
    algebra_problem, algebra_solution = generate_algebra_problem(params, subkey)
    print(f"Algebra Problem: {algebra_problem}")
    print(f"Algebra Solution: {algebra_solution}")

    rng_key, subkey = jax.random.split(rng_key)
    calculus_problem, calculus_solution = generate_calculus_problem(params, subkey)
    print(f"Calculus Problem: {calculus_problem}")
    print(f"Calculus Solution: {calculus_solution}")