import sympy as sp

def solve_algebra_problem(problem):
    """
    Solves a given algebraic problem using SymPy.
    Returns the solution as a list of sympy expressions or None if an error occurs.
    """
    try:
        solution = sp.solve(problem)
        return solution
    except Exception as e:
        print(f"Error solving algebra problem: {e}")
        return None

def solve_calculus_problem(problem):
    """
    Solves a given calculus problem using SymPy.
    Returns the solution as a sympy expression or None if an error occurs.
    """
    try:
        solution = sp.integrate(problem.rhs, problem.lhs)
        return [solution]  # Return as a list for consistency
    except Exception as e:
        print(f"Error solving calculus problem: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    from problem_generation import generate_algebra_problem, generate_calculus_problem

    algebra_problem, _ = generate_algebra_problem()
    algebra_solution = solve_algebra_problem(algebra_problem)
    if algebra_solution is not None:
        print(f"Algebra Problem: {algebra_problem}")
        print(f"Algebra Solution: {algebra_solution}")
    else:
        print("Failed to solve algebra problem.")

    calculus_problem, _ = generate_calculus_problem()
    calculus_solution = solve_calculus_problem(calculus_problem)
    if calculus_solution is not None:
        print(f"Calculus Problem: {calculus_problem}")
        print(f"Calculus Solution: {calculus_solution}")
    else:
        print("Failed to solve calculus problem.")
