import unittest
from src.problem_generation import generate_algebra_problem, generate_calculus_problem
from src.problem_solving import solve_algebra_problem, solve_calculus_problem
import sympy as sp

class TestProblemSolving(unittest.TestCase):

    def test_solve_algebra_problem(self):
        problem, expected_solution = generate_algebra_problem()
        solution = solve_algebra_problem(problem)
        self.assertIsInstance(solution, list)
        self.assertEqual(solution, expected_solution)

    def test_solve_calculus_problem(self):
        problem, expected_solution = generate_calculus_problem()
        solution = solve_calculus_problem(problem)
        self.assertIsInstance(solution, list)
        self.assertEqual(solution[0], expected_solution)

if __name__ == "__main__":
    unittest.main()
