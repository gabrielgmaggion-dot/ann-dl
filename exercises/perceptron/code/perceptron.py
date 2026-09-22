"""Perceptron exercise: reproducible experiments and Figures 1--6.

The classifier, activation, update rule, and training loop are implemented
directly with NumPy. No model from scikit-learn is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = BASE_DIR / "figures"


@dataclass
class TrainingResult:
    """Values saved at the end of a perceptron training run."""

    weights: np.ndarray
    bias: float
    epochs: int
    accuracy_history: list[float]
    updates_history: list[int]
    pocket_weights: np.ndarray | None = None
    pocket_bias: float | None = None
    pocket_accuracy: float | None = None
    pocket_epoch: int | None = None
    pocket_history: list[float] | None = None


class Perceptron:
    """Binary perceptron for labels in {0, 1}, implemented from scratch."""

    def __init__(self, learning_rate: float, initial_weights: np.ndarray):
        self.learning_rate = learning_rate
        self.weights = np.asarray(initial_weights, dtype=float).copy()
        self.bias = 0.0

    @staticmethod
    def step(score: np.ndarray | float) -> np.ndarray | int:
        """Return class 1 when the linear score is non-negative."""
        output = (np.asarray(score) >= 0.0).astype(int)
        return int(output) if output.ndim == 0 else output

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict all rows in X using the current parameters."""
        return self.step(X @ self.weights + self.bias)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the fraction of correct predictions."""
        return float(np.mean(self.predict(X) == y))

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_epochs: int = 100,
        keep_pocket: bool = False,
    ) -> TrainingResult:
        """Train in fixed sample order and optionally keep best-so-far weights.

        When ``keep_pocket`` is true, the full-data accuracy is measured after
        each mistaken-sample update. A copy is kept only when that accuracy is
        strictly higher than the previous best.
        """
        accuracy_history: list[float] = []
        updates_history: list[int] = []

        pocket_weights = self.weights.copy()
        pocket_bias = self.bias
        pocket_accuracy = self.accuracy(X, y)
        pocket_epoch = 0
        pocket_history: list[float] = []

        for epoch in range(1, max_epochs + 1):
            updates = 0

            for features, target in zip(X, y):
                prediction = self.step(self.weights @ features + self.bias)
                error = int(target) - prediction

                if error != 0:
                    self.weights += self.learning_rate * error * features
                    self.bias += self.learning_rate * error
                    updates += 1

                    if keep_pocket:
                        updated_accuracy = self.accuracy(X, y)
                        if updated_accuracy > pocket_accuracy:
                            pocket_accuracy = updated_accuracy
                            pocket_weights = self.weights.copy()
                            pocket_bias = self.bias
                            pocket_epoch = epoch

            current_accuracy = self.accuracy(X, y)
            accuracy_history.append(current_accuracy)
            updates_history.append(updates)

            if keep_pocket:
                pocket_history.append(pocket_accuracy)

            # A full pass with no update is the stated convergence criterion.
            if updates == 0:
                break

        return TrainingResult(
            weights=self.weights.copy(),
            bias=self.bias,
            epochs=len(accuracy_history),
            accuracy_history=accuracy_history,
            updates_history=updates_history,
            pocket_weights=pocket_weights if keep_pocket else None,
            pocket_bias=pocket_bias if keep_pocket else None,
            pocket_accuracy=pocket_accuracy if keep_pocket else None,
            pocket_epoch=pocket_epoch if keep_pocket else None,
            pocket_history=pocket_history if keep_pocket else None,
        )


def generate_two_classes(
    rng: np.random.Generator,
    mean_class_0: list[float],
    mean_class_1: list[float],
    covariance: np.ndarray,
    samples_per_class: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate and stack class 0 followed by class 1."""
    class_0 = rng.multivariate_normal(mean_class_0, covariance, samples_per_class)
    class_1 = rng.multivariate_normal(mean_class_1, covariance, samples_per_class)
    X = np.vstack((class_0, class_1))
    y = np.concatenate(
        (np.zeros(samples_per_class, dtype=int), np.ones(samples_per_class, dtype=int))
    )
    return X, y


def set_plot_style() -> None:
    """Use one readable visual style for all six figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 130,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": True,
        }
    )


def scatter_classes(ax: plt.Axes, X: np.ndarray, y: np.ndarray) -> None:
    """Draw the two classes with consistent colors and labels."""
    colors = ("#2f6fbb", "#ef8a24")
    for class_value, color in enumerate(colors):
        mask = y == class_value
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=16,
            alpha=0.58,
            color=color,
            edgecolors="none",
            label=f"Class {class_value}",
            rasterized=True,
        )


def draw_boundary(
    ax: plt.Axes,
    weights: np.ndarray,
    bias: float,
    label: str,
    color: str,
    linestyle: str = "-",
) -> None:
    """Draw w.x + b = 0 inside the current axes limits."""
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    if abs(weights[1]) > 1e-12:
        x_values = np.asarray(x_limits)
        y_values = -(weights[0] * x_values + bias) / weights[1]
        ax.plot(x_values, y_values, color=color, lw=2.2, ls=linestyle, label=label)
    elif abs(weights[0]) > 1e-12:
        x_value = -bias / weights[0]
        ax.axvline(x_value, color=color, lw=2.2, ls=linestyle, label=label)

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)


def mark_mistakes(
    ax: plt.Axes,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    label: str = "Misclassified",
) -> int:
    """Circle points misclassified by the supplied parameters."""
    predictions = Perceptron.step(X @ weights + bias)
    mistakes = predictions != y
    ax.scatter(
        X[mistakes, 0],
        X[mistakes, 1],
        s=46,
        facecolors="none",
        edgecolors="#c62828",
        linewidths=1.2,
        marker="o",
        label=label,
        zorder=4,
        rasterized=True,
    )
    return int(np.sum(mistakes))


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a tightly cropped PNG and release its memory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def run_experiments() -> dict[str, object]:
    """Run both exercises with the one required RNG and create all figures."""
    rng = np.random.default_rng(42)
    set_plot_style()

    # Exercise 1: separable data.
    covariance_1 = np.array([[0.5, 0.0], [0.0, 0.5]])
    X_1, y_1 = generate_two_classes(
        rng, [1.5, 1.5], [5.0, 5.0], covariance_1
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    scatter_classes(ax, X_1, y_1)
    ax.set(title="Figure 1 — Separable data", xlabel="$x_1$", ylabel="$x_2$")
    ax.legend()
    save_figure(fig, "figure-1-separable-data.svg")

    # Both learning-rate runs start from the same non-zero vector, so eta is
    # the only experimental change in Exercise 1D.
    initial_weights_1 = rng.normal(0.0, 0.01, size=2)
    model_001 = Perceptron(learning_rate=0.01, initial_weights=initial_weights_1)
    result_001 = model_001.train(X_1, y_1, max_epochs=100)

    model_1 = Perceptron(learning_rate=1.0, initial_weights=initial_weights_1)
    result_1 = model_1.train(X_1, y_1, max_epochs=100)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    scatter_classes(ax, X_1, y_1)
    draw_boundary(
        ax,
        result_001.weights,
        result_001.bias,
        "Decision boundary",
        color="#222222",
    )
    mark_mistakes(ax, X_1, y_1, result_001.weights, result_001.bias)
    ax.set(
        title="Figure 2 — Perceptron boundary on separable data",
        xlabel="$x_1$",
        ylabel="$x_2$",
    )
    ax.legend()
    save_figure(fig, "figure-2-separable-boundary.svg")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    epochs_1 = np.arange(1, result_001.epochs + 1)
    ax.plot(
        epochs_1,
        np.asarray(result_001.accuracy_history) * 100,
        color="#2f6fbb",
        marker="o",
        lw=2,
        label="Current weights",
    )
    ax.set(
        title="Figure 3 — Accuracy by epoch on separable data",
        xlabel="Epoch",
        ylabel="Accuracy (%)",
        xticks=epochs_1,
    )
    # Keep the complete learning trajectory visible, including the early epochs.
    ax.set_ylim(49, 100.75)
    ax.legend()
    save_figure(fig, "figure-3-separable-accuracy.svg")

    # Exercise 2: overlapping data. The same rng continues from Exercise 1.
    covariance_2 = np.array([[1.5, 0.0], [0.0, 1.5]])
    X_2, y_2 = generate_two_classes(
        rng, [3.0, 3.0], [4.0, 4.0], covariance_2
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    scatter_classes(ax, X_2, y_2)
    ax.set(title="Figure 4 — Overlapping data", xlabel="$x_1$", ylabel="$x_2$")
    ax.legend()
    save_figure(fig, "figure-4-overlapping-data.svg")

    initial_weights_2 = rng.normal(0.0, 0.01, size=2)
    model_2 = Perceptron(learning_rate=0.01, initial_weights=initial_weights_2)
    result_2 = model_2.train(X_2, y_2, max_epochs=100, keep_pocket=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharex=True, sharey=True)
    parameter_sets = (
        ("Final iterate", result_2.weights, result_2.bias),
        ("Pocket (best-so-far)", result_2.pocket_weights, result_2.pocket_bias),
    )
    for ax, (name, weights, bias) in zip(axes, parameter_sets):
        assert weights is not None and bias is not None
        scatter_classes(ax, X_2, y_2)
        draw_boundary(ax, weights, bias, f"{name} boundary", color="#222222")
        mistakes = mark_mistakes(ax, X_2, y_2, weights, bias)
        ax.set(
            title=f"{name}: {mistakes} mistakes",
            xlabel="$x_1$",
            ylabel="$x_2$",
        )
        ax.legend(fontsize=8)
    fig.suptitle("Figure 5 — Final and pocket boundaries on overlapping data", y=1.02)
    save_figure(fig, "figure-5-overlapping-boundaries.svg")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    epochs_2 = np.arange(1, result_2.epochs + 1)
    ax.plot(
        epochs_2,
        np.asarray(result_2.accuracy_history) * 100,
        color="#ef8a24",
        lw=1.6,
        label="Current weights",
    )
    assert result_2.pocket_history is not None
    ax.plot(
        epochs_2,
        np.asarray(result_2.pocket_history) * 100,
        color="#2f6fbb",
        lw=2.2,
        label="Pocket (best-so-far)",
    )
    ax.set(
        title="Figure 6 — Current and pocket accuracy by epoch",
        xlabel="Epoch",
        ylabel="Accuracy (%)",
    )
    ax.legend()
    save_figure(fig, "figure-6-overlapping-accuracy.svg")

    final_accuracy_1 = float(result_001.accuracy_history[-1])
    final_accuracy_1_eta_1 = float(result_1.accuracy_history[-1])
    final_accuracy_2 = float(result_2.accuracy_history[-1])

    return {
        "exercise_1": {
            "initial_weights": initial_weights_1,
            "eta_0.01": result_001,
            "eta_1.0": result_1,
            "final_accuracy_eta_0.01": final_accuracy_1,
            "final_accuracy_eta_1.0": final_accuracy_1_eta_1,
            "direction_eta_0.01": result_001.weights
            / np.linalg.norm(result_001.weights),
            "direction_eta_1.0": result_1.weights / np.linalg.norm(result_1.weights),
        },
        "exercise_2": {
            "initial_weights": initial_weights_2,
            "result": result_2,
            "final_accuracy": final_accuracy_2,
        },
    }


def print_results(results: dict[str, object]) -> None:
    """Print the exact values used in the written report."""
    exercise_1 = results["exercise_1"]
    exercise_2 = results["exercise_2"]
    assert isinstance(exercise_1, dict) and isinstance(exercise_2, dict)

    result_001 = exercise_1["eta_0.01"]
    result_1 = exercise_1["eta_1.0"]
    result_2 = exercise_2["result"]
    assert isinstance(result_001, TrainingResult)
    assert isinstance(result_1, TrainingResult)
    assert isinstance(result_2, TrainingResult)

    print("Exercise 1")
    print(f"initial w = {exercise_1['initial_weights']}")
    print(
        f"eta=0.01: w={result_001.weights}, b={result_001.bias:.8f}, "
        f"epochs={result_001.epochs}, accuracy={exercise_1['final_accuracy_eta_0.01']:.6f}"
    )
    print(f"eta=0.01 direction = {exercise_1['direction_eta_0.01']}")
    print(f"updates by epoch = {result_001.updates_history}")
    print(
        f"eta=1.0: w={result_1.weights}, b={result_1.bias:.8f}, "
        f"epochs={result_1.epochs}, accuracy={exercise_1['final_accuracy_eta_1.0']:.6f}"
    )
    print(f"eta=1.0 direction = {exercise_1['direction_eta_1.0']}")

    print("\nExercise 2")
    print(f"initial w = {exercise_2['initial_weights']}")
    print(
        f"final: w={result_2.weights}, b={result_2.bias:.8f}, "
        f"epochs={result_2.epochs}, accuracy={exercise_2['final_accuracy']:.6f}"
    )
    print(
        f"pocket: w={result_2.pocket_weights}, b={result_2.pocket_bias:.8f}, "
        f"epoch={result_2.pocket_epoch}, accuracy={result_2.pocket_accuracy:.6f}"
    )
    print(f"last 10 epoch accuracies = {result_2.accuracy_history[-10:]}")


if __name__ == "__main__":
    experiment_results = run_experiments()
    print_results(experiment_results)
