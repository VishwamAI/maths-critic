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
    Handles potential errors and ensures only valid ASCII characters are used.
    """
    decoded = []
    for i in output.flatten():
        try:
            code = int(i)
            if 32 <= code <= 126:  # printable ASCII range
                decoded.append(chr(code))
            else:
                decoded.append(' ')  # replace non-printable characters with space
        except ValueError:
            # If conversion to int fails, skip this value
            continue
    return ''.join(decoded).strip()

def generate_algebra_problem(params, rng_key, expression=None):
    """
    Generates a simple algebraic problem of the form ax + b = c and solves it,
    or uses a provided expression.
    Returns the problem as a string and the solution.
    """
    if expression:
        # Parse the expression and return it directly
        return expression, eval(expression)

    # Generate random coefficients
    a, b, c = jax.random.randint(rng_key, (3,), 1, 10)  # Ensure 'a' is not zero
    problem = f"{a}x + {b} = {c}"
    # Solve the equation: ax + b = c
    x = (c - b) / a
    solution = f"x = {x:.2f}"
    return problem, solution

def generate_calculus_problem(model, params, rng_key):
    """
    Generates a simple calculus problem of the form f(x) = integral(g(x)) using NextGenJAX model.
    Returns the problem as a string and the solution.
    """
    rng_key, subkey = jax.random.split(rng_key)
    a, b, c = jax.random.randint(subkey, (3,), 0, 10)
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
    algebra_problem, algebra_solution = generate_algebra_problem(params, subkey)
    print(f"Algebra Problem: {algebra_problem}")
    print(f"Algebra Solution: {algebra_solution}")

    rng_key, subkey = jax.random.split(rng_key)
    calculus_problem, calculus_solution = generate_calculus_problem(model, params, subkey)
    print(f"Calculus Problem: {calculus_problem}")
    print(f"Calculus Solution: {calculus_solution}")