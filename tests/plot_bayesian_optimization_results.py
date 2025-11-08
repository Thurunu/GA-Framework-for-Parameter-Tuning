import matplotlib.pyplot as plt  # For visualization of optimization results
import json  # For loading optimization results from JSON file


def plot_optimization_history(result):
    """
    Visualizes the optimization process using matplotlib.
    Plots:
    - Score vs. Iteration (line chart)
    - Best score progression (line chart)
    """
    scores = result.scores
    best_scores = []
    current_best = float('-inf')
    for s in scores:
        current_best = max(current_best, s)
        best_scores.append(current_best)

    plt.figure(figsize=(10, 6))
    plt.plot(scores, label='Score per Iteration', marker='o')
    plt.plot(best_scores, label='Best Score Progression', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Bayesian Optimization Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def load_optimization_results(filename):
    """
    Load optimization results from a JSON file.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    # Convert back to OptimizationResult-like structure
    class MockResult:
        def __init__(self, data):
            self.scores = [entry["score"] for entry in data["evaluation_history"]]
            self.evaluation_history = data['evaluation_history']
            self.best_score = data['best_result']["score"]
            self.best_parameters = data['best_result']["parameters"]
            self.iteration_count = len(data['evaluation_history'])
            # self.convergence_reached = data['convergence_reached']
            # self.optimization_time = data['optimization_time']
    return MockResult(data)

if __name__ == "__main__":
    # Load results from the JSON file
    result = load_optimization_results('tests/test_bayesian_optimization_results.json')

    plot_optimization_history(result)
