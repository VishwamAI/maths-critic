# Advanced Critical Math-Solving Agent

## Introduction
The Advanced Critical Math-Solving Agent is an AI-based system designed to generate, solve, and learn from complex mathematical problems. It covers a wide range of mathematical areas, including algebra, calculus, statistics, differential equations, and numerical analysis. The agent leverages advanced reinforcement learning techniques to improve its problem-solving capabilities over time.

## Features
- **Problem Generation**: Generates algebra and calculus problems.
- **Problem Solving**: Solves algebra and calculus problems using SymPy.
- **Reinforcement Learning**: Utilizes a Q-learning algorithm to enhance problem-solving strategies.

## Installation
To set up the math-solving agent, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/VishwamAI/maths-critic.git
   cd maths-critic
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Generating Problems
To generate algebra and calculus problems, use the functions provided in the `problem_generation.py` module:
```python
from src.problem_generation import generate_algebra_problem, generate_calculus_problem

algebra_problem, algebra_solution = generate_algebra_problem()
print(f"Algebra Problem: {algebra_problem}")
print(f"Algebra Solution: {algebra_solution}")

calculus_problem, calculus_solution = generate_calculus_problem()
print(f"Calculus Problem: {calculus_problem}")
print(f"Calculus Solution: {calculus_solution}")
```

### Solving Problems
To solve algebra and calculus problems, use the functions provided in the `problem_solving.py` module:
```python
from src.problem_solving import solve_algebra_problem, solve_calculus_problem

algebra_solution = solve_algebra_problem(algebra_problem)
print(f"Algebra Solution: {algebra_solution}")

calculus_solution = solve_calculus_problem(calculus_problem)
print(f"Calculus Solution: {calculus_solution}")
```

### Reinforcement Learning
To run the reinforcement learning loop and improve the agent's problem-solving strategies, use the `learning_module.py` module:
```python
from src.learning_module import LearningModule

learning_module = LearningModule()
learning_module.reinforcement_learning_loop()
```

## Limitations
- The current implementation focuses on algebra and calculus problems. Future updates will expand the range of mathematical areas covered.
- The reinforcement learning module is in its initial stages and may require further tuning for optimal performance.

## Contributing
Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`.
3. Make your changes and commit them: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature-branch`.
5. Create a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
