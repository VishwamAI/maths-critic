from src.problem_generation import generate_algebra_problem, generate_calculus_problem
from src.problem_solving import solve_algebra_problem, solve_calculus_problem
from src.learning_module import LearningModule

def main():
    # Generate an algebra problem
    algebra_problem, algebra_solution = generate_algebra_problem()
    print("Generated Algebra Problem:", algebra_problem)
    print("Expected Solution:", algebra_solution)

    # Solve the algebra problem
    solved_algebra = solve_algebra_problem(algebra_problem)
    print("Solved Algebra Problem:", solved_algebra)

    # Generate a calculus problem
    calculus_problem, calculus_solution = generate_calculus_problem()
    print("Generated Calculus Problem:", calculus_problem)
    print("Expected Solution:", calculus_solution)

    # Solve the calculus problem
    solved_calculus = solve_calculus_problem(calculus_problem)
    print("Solved Calculus Problem:", solved_calculus)

    # Initialize the learning module
    learning_module = LearningModule()

    # Run the reinforcement learning loop
    learning_module.run_learning_loop(episodes=10)
    print("Reinforcement learning completed.")

if __name__ == "__main__":
    main()
