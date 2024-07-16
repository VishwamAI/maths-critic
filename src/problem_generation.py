import random
import jax
import jax.numpy as jnp
from nextgenjax.model import create_model

def initialize_nextgenjax_model():
    """
    Initialize and return the NextGenJAX model with predefined hyperparameters.
    """
    return create_model(num_layers=6, hidden_size=512, num_heads=8, dropout_rate=0.1)

def encode_problem(problem: str) -> jnp.ndarray:
    """
    Convert problem string to a numerical representation.
    Ensures output shape is (1, 512) to match model expectations.
    """
    encoded = jnp.array([ord(c) for c in problem])
    padded = jnp.pad(encoded, (0, max(0, 512 - len(encoded))))
    return padded.reshape(1, 512)

def decode_solution(output: jnp.ndarray) -> str:
    """
    Convert model output to a solution string.
    This is a placeholder implementation; adjust based on your specific decoding scheme.
    """
    return ''.join([chr(int(i)) for i in output.flatten()])

def generate_algebra_problem(model, params, rng_key):
    """
    Generates a simple algebraic problem of the form ax + b = c using NextGenJAX model.
    Returns the problem as a string and the solution.
    """
    a, b, c = jax.random.randint(rng_key, (3,), 0, 10)
    problem = f"{a}x + {b} = {c}"
    encoded_problem = encode_problem(problem)
    output = model.apply(params, rng_key, encoded_problem)
    solution = decode_solution(output)
    return problem, solution

def generate_calculus_problem(model, params, rng_key):
    """
    Generates a simple calculus problem of the form f(x) = integral(g(x)) using NextGenJAX model.
    Returns the problem as a string and the solution.
    """
    a, b, c = jax.random.randint(rng_key, (3,), 0, 10)
    g = f"{a}x^2 + {b}x + {c}"
    problem = f"f(x) = integral({g})"
    encoded_problem = encode_problem(problem)
    output = model.apply(params, rng_key, encoded_problem)
    solution = decode_solution(output)
    return problem, solution

if __name__ == "__main__":
    model = initialize_nextgenjax_model()
    rng_key = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, 512))  # Matches the model's input expectations
    params = model.init(rng_key, dummy_input)

    rng_key, subkey = jax.random.split(rng_key)
    algebra_problem, algebra_solution = generate_algebra_problem(model, params, subkey)
    print(f"Algebra Problem: {algebra_problem}")
    print(f"Algebra Solution: {algebra_solution}")

    rng_key, subkey = jax.random.split(rng_key)
    calculus_problem, calculus_solution = generate_calculus_problem(model, params, subkey)
    print(f"Calculus Problem: {calculus_problem}")
    print(f"Calculus Solution: {calculus_solution}")