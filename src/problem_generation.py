import random
import sympy as sp

def generate_algebra_problem():
    """
    Generates a simple algebraic problem of the form ax + b = c.
    Returns the problem as a string and the solution as a sympy expression.
    """
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    c = random.randint(1, 10)

    x = sp.symbols('x')
    problem = sp.Eq(a * x + b, c)
    solution = sp.solve(problem, x)

    return problem, solution

def generate_calculus_problem():
    """
    Generates a simple calculus problem of the form f(x) = integral(g(x)).
    Returns the problem as a string and the solution as a sympy expression.
    """
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    c = random.randint(1, 10)

    x = sp.symbols('x')
    g = a * x**2 + b * x + c
    f = sp.integrate(g, x)
    problem = sp.Eq(sp.Function('f')(x), f)

    return problem, f

if __name__ == "__main__":
    algebra_problem, algebra_solution = generate_algebra_problem()
    print(f"Algebra Problem: {algebra_problem}")
    print(f"Algebra Solution: {algebra_solution}")

    calculus_problem, calculus_solution = generate_calculus_problem()
    print(f"Calculus Problem: {calculus_problem}")
    print(f"Calculus Solution: {calculus_solution}")
