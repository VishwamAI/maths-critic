import unittest
from src.problem_generation import generate_algebra_problem, generate_calculus_problem
import sympy as sp

class TestProblemGeneration(unittest.TestCase):

    def test_generate_algebra_problem(self):
        problem, solution = generate_algebra_problem()
        self.assertIsInstance(problem, sp.Equality)
        self.assertIsInstance(solution, list)
        self.assertTrue(all(isinstance(sol, sp.Basic) for sol in solution))

    def test_generate_calculus_problem(self):
        problem, solution = generate_calculus_problem()
        self.assertIsInstance(problem, sp.Equality)
        self.assertIsInstance(solution, sp.Basic)

if __name__ == "__main__":
    unittest.main()
