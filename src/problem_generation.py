import random
from nextgenjax import NextGenJAX

def initialize_nextgenjax_model():
    """
    Initialize and return the NextGenJAX model.
    """
    # TODO: Replace with actual NextGenJAX initialization
    return NextGenJAX()

model = initialize_nextgenjax_model()

def generate_algebra_problem():
    """
    Generates a simple algebraic problem of the form ax + b = c using NextGenJAX.
    Returns the problem as a string and the solution.
    """
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    c = random.randint(1, 10)

    problem = f"{a}x + {b} = {c}"
    solution = model.solve_algebra(problem)

    return problem, solution

def generate_calculus_problem():
    """
    Generates a simple calculus problem of the form f(x) = integral(g(x)) using NextGenJAX.
    Returns the problem as a string and the solution.
    """
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    c = random.randint(1, 10)

    g = f"{a}x^2 + {b}x + {c}"
    problem = f"f(x) = integral({g})"
    solution = model.solve_calculus(problem)

    return problem, solution

if __name__ == "__main__":
    algebra_problem, algebra_solution = generate_algebra_problem()
    print(f"Algebra Problem: {algebra_problem}")
    print(f"Algebra Solution: {algebra_solution}")

    calculus_problem, calculus_solution = generate_calculus_problem()
    print(f"Calculus Problem: {calculus_problem}")
    print(f"Calculus Solution: {calculus_solution}")