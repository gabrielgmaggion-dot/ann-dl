"""MLP do zero: execute este arquivo para reproduzir resultados e Figuras 1–6.

O scikit-learn só gera/separa dados, calcula a matriz de confusão e fornece
as regressões logísticas de referência expressamente permitidas no enunciado.
Forward, perdas, backpropagation e gradiente descendente usam somente NumPy.
"""
from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import sklearn

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
rng = np.random.default_rng(42)  # Único Generator, passado a todas as redes.
COLORS = ["#3368a8", "#dd8039", "#5c9983"]


class MLP:
    """Qualquer sequência de camadas; tanh nas ocultas e saída configurável."""

    def __init__(self, layer_sizes, rng, output="binary"):
        self.layer_sizes = list(layer_sizes)
        self.output = output
        if output not in ("binary", "multiclass"):
            raise ValueError("Saída deve ser binary ou multiclass.")
        if output == "binary" and layer_sizes[-1] != 1:
            raise ValueError("Classificação binária requer uma saída.")
        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Xavier uniforme: escala adequada à tanh; vieses inicialmente zero.
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(rng.uniform(-limit, limit, (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

    @staticmethod
    def sigmoid(z):
        # Formulação estável sem overflow, inclusive para logits extremos.
        result = np.empty_like(z)
        positive = z >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        result[~positive] = exp_z / (1.0 + exp_z)
        return result

    def forward(self, X):
        activations = [X]
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            activations.append(np.tanh(activations[-1] @ W + b))
        logits = activations[-1] @ self.weights[-1] + self.biases[-1]
        if self.output == "binary":
            probabilities = self.sigmoid(logits)
        else:
            shifted = logits - logits.max(axis=1, keepdims=True)
            exp_z = np.exp(shifted)
            probabilities = exp_z / exp_z.sum(axis=1, keepdims=True)
        activations.append(probabilities)
        return probabilities, (activations, logits)

    def targets(self, y):
        if self.output == "binary":
            return y.reshape(-1, 1).astype(float)
        return np.eye(self.layer_sizes[-1])[y.astype(int)]

    def loss(self, logits, Y):
        if self.output == "binary":
            # BCE = log(1 + exp(z)) - y*z, calculada de modo estável.
            return float(np.mean(np.logaddexp(0.0, logits) - Y * logits))
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return float(-np.mean(np.sum(Y * log_probs, axis=1)))

    def backward(self, cache, Y):
        activations, _ = cache
        # Sigmoid+BCE e softmax+CCE possuem o mesmo delta: (P-Y)/N.
        delta = (activations[-1] - Y) / len(Y)
        dW, db = [None] * len(self.weights), [None] * len(self.biases)
        for layer in range(len(self.weights) - 1, -1, -1):
            dW[layer] = activations[layer].T @ delta
            db[layer] = delta.sum(axis=0, keepdims=True)
            if layer > 0:
                delta = (delta @ self.weights[layer].T) * (1.0 - activations[layer] ** 2)
        return dW, db

    def update(self, gradients, learning_rate):
        dW, db = gradients
        # Todos os gradientes já foram calculados com os parâmetros antigos.
        for layer in range(len(self.weights)):
            self.weights[layer] -= learning_rate * dW[layer]
            self.biases[layer] -= learning_rate * db[layer]

    def fit(self, X, y, learning_rate, epochs):
        Y = self.targets(y)
        _, cache = self.forward(X)
        history = [self.loss(cache[1], Y)]  # Época zero: antes das atualizações.
        for _ in range(epochs):
            self.update(self.backward(cache, Y), learning_rate)
            _, cache = self.forward(X)
            history.append(self.loss(cache[1], Y))
        return np.asarray(history)

    def predict(self, X):
        probabilities, _ = self.forward(X)
        if self.output == "binary":
            return (probabilities[:, 0] >= 0.5).astype(int)
        return probabilities.argmax(axis=1)

    @property
    def parameter_count(self):
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))


def reset_seed(seed):
    """Reinicia o MESMO rng apenas para as repetições pareadas exigidas."""
    rng.bit_generator.state = np.random.PCG64(seed).state


def standardize_train_test(X_train, X_test):
    """Estima média/desvio exclusivamente no treino, sem vazamento do teste."""
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def save_figure(fig, number):
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig{number}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def loss_figure(history, number, title, classes):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(len(history)), history, color=COLORS[0], label=f"Treino — classes {classes}")
    ax.set(title=f"Figura {number} — {title}", xlabel="Época (0 = inicial)", ylabel="Entropia cruzada média")
    ax.legend()
    ax.grid(alpha=.2)
    save_figure(fig, number)


def check_gradients():
    """Diferenças centrais verificam todas as entradas em três redes pequenas."""
    saved_state = rng.bit_generator.state
    X = np.array([[.2, -.4], [.7, .3], [-.5, .8]])
    checks = []
    for sizes, output, y in [([2, 3, 1], "binary", np.array([0, 1, 0])),
                              ([2, 3, 3], "multiclass", np.array([0, 1, 2])),
                              ([2, 3, 2, 3], "multiclass", np.array([0, 1, 2]))]:
        model = MLP(sizes, rng, output)
        Y = model.targets(y)
        _, cache = model.forward(X)
        dW, db = model.backward(cache, Y)
        max_error = 0.0
        for parameter, gradient in zip(model.weights + model.biases, dW + db):
            for index in np.ndindex(parameter.shape):
                original, epsilon = parameter[index], 1e-6
                parameter[index] = original + epsilon
                positive = model.loss(model.forward(X)[1][1], Y)
                parameter[index] = original - epsilon
                negative = model.loss(model.forward(X)[1][1], Y)
                parameter[index] = original
                numerical = (positive - negative) / (2 * epsilon)
                max_error = max(max_error, abs(numerical - gradient[index]))
        assert max_error < 1e-7, (sizes, max_error)
        checks.append({"architecture": sizes, "max_absolute_error": max_error})
    rng.bit_generator.state = saved_state
    return checks


def main():
    reset_seed(42)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none", "svg.hashsalt": "mlp42"})
    results = {"versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
                           "scikit-learn": sklearn.__version__}, "gradient_checks": check_gradients()}

    # Exercício 2: gerador e divisão exatamente como no enunciado.
    centers = np.array([[0., 0.], [3., 3.], [-3., -3.]])
    X, c = make_blobs(n_samples=[500, 250, 250], centers=centers,
                      cluster_std=1.2, random_state=42)
    y = (c > 0).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    Xtrs, Xtes, mean2, std2 = standardize_train_test(Xtr, Xte)
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls in (0, 1):
        ax.scatter(X[y == cls, 0], X[y == cls, 1], s=15, alpha=.65, color=COLORS[cls], label=f"Classe {cls}")
    ax.set(title="Figura 1 — Dados binários: três agrupamentos", xlabel="$x_1$", ylabel="$x_2$")
    ax.legend()
    ax.grid(alpha=.15)
    save_figure(fig, 1)
    baseline2 = LogisticRegression(max_iter=2000, random_state=42).fit(Xtrs, ytr)
    model2 = MLP([2, 8, 1], rng, "binary")
    history2 = model2.fit(Xtrs, ytr, learning_rate=.1, epochs=3000)
    pred2 = model2.predict(Xtes)
    results["exercise2"] = {"architecture": model2.layer_sizes, "learning_rate": .1, "epochs": 3000,
                            "parameters": model2.parameter_count, "baseline_accuracy": accuracy(yte, baseline2.predict(Xtes)),
                            "test_accuracy": accuracy(yte, pred2), "correct": int((yte == pred2).sum()),
                            "initial_loss": float(history2[0]), "final_loss": float(history2[-1]),
                            "train_class_counts": np.bincount(ytr).tolist(), "test_class_counts": np.bincount(yte).tolist()}
    loss_figure(history2, 2, "Perda do MLP binário", "0 e 1")
    gx = np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 320)
    gy = np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 320)
    xx, yy = np.meshgrid(gx, gy)
    grid = (np.c_[xx.ravel(), yy.ravel()] - mean2) / std2
    probabilities, _ = model2.forward(grid)
    zz = probabilities.reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, zz >= .5, levels=[-.5, .5, 1.5], cmap=ListedColormap(COLORS[:2]), alpha=.17)
    ax.contour(xx, yy, zz, levels=[.5], colors="#303030", linewidths=1.3)
    for cls in (0, 1):
        ax.scatter(Xte[yte == cls, 0], Xte[yte == cls, 1], color=COLORS[cls], s=28,
                   edgecolors="white", linewidths=.35, label=f"Classe {cls} — teste")
    ax.set(title="Figura 3 — Fronteira do MLP e pontos de teste", xlabel="$x_1$", ylabel="$x_2$")
    ax.legend()
    save_figure(fig, 3)

    # Exercícios 3 e 4: dados/split permanecem fixos nas três sementes de pesos.
    X, y = make_classification(n_samples=1500, n_features=4, n_informative=4,
                               n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=2, class_sep=1.0, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    Xtrs, Xtes, _, _ = standardize_train_test(Xtr, Xte)
    baseline3 = LogisticRegression(max_iter=2000, random_state=42).fit(Xtrs, ytr)
    results["multiclass_data"] = {"train_class_counts": np.bincount(ytr).tolist(),
                                  "test_class_counts": np.bincount(yte).tolist(),
                                  "baseline_accuracy": accuracy(yte, baseline3.predict(Xtes))}
    runs, histories, models = [], {}, {}
    for seed in (42, 43, 44):
        for exercise, sizes in [(3, [4, 16, 3]), (4, [4, 16, 16, 3])]:
            reset_seed(seed)
            model = MLP(sizes, rng, "multiclass")
            history = model.fit(Xtrs, ytr, learning_rate=.1, epochs=3000)
            pred = model.predict(Xtes)
            row = {"exercise": exercise, "seed": seed, "architecture": sizes,
                   "parameters": model.parameter_count, "epochs": 3000, "learning_rate": .1,
                   "initial_loss": float(history[0]), "final_loss": float(history[-1]),
                   "train_accuracy": accuracy(ytr, model.predict(Xtrs)), "test_accuracy": accuracy(yte, pred),
                   "correct": int((pred == yte).sum())}
            runs.append(row)
            if seed == 42:
                histories[exercise], models[exercise] = history, model
                results[f"exercise{exercise}"] = row.copy()
    cm = confusion_matrix(yte, models[3].predict(Xtes), labels=[0, 1, 2])
    results["exercise3"]["confusion_matrix"] = cm.tolist()
    pair_counts = [(int(cm[i, j] + cm[j, i]), i, j) for i in range(3) for j in range(i + 1, 3)]
    largest, first, second = max(pair_counts)
    results["exercise3"]["most_confused_pair"] = [first, second]
    results["exercise3"]["pair_errors"] = largest
    results["runs"] = runs
    results["means"] = {}
    for ex in (3, 4):
        subset = [row for row in runs if row["exercise"] == ex]
        results["means"][str(ex)] = {key: float(np.mean([row[key] for row in subset]))
                                      for key in ("test_accuracy", "final_loss", "train_accuracy")}
        results["means"][str(ex)]["test_std"] = float(np.std([row["test_accuracy"] for row in subset], ddof=1))
    results["mean_accuracy_difference_pp"] = 100 * (results["means"]["4"]["test_accuracy"] - results["means"]["3"]["test_accuracy"])
    loss_figure(histories[3], 4, "Perda do MLP multiclasse", "0, 1 e 2")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "#182739")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=["Classe 0", "Classe 1", "Classe 2"],
           yticklabels=["Classe 0", "Classe 1", "Classe 2"], xlabel="Classe predita", ylabel="Classe verdadeira",
           title="Figura 5 — Matriz de confusão: MLP 4–16–3")
    fig.colorbar(im, ax=ax, label="Número de observações")
    save_figure(fig, 5)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(histories[3], color=COLORS[0], label="1 camada oculta — classes 0, 1, 2")
    ax.plot(histories[4], color=COLORS[1], label="2 camadas ocultas — classes 0, 1, 2")
    ax.set(title="Figura 6 — Profundidade e perda de treino (semente 42)", xlabel="Época (0 = inicial)", ylabel="Entropia cruzada média")
    ax.legend()
    ax.grid(alpha=.2)
    save_figure(fig, 6)
    (ROOT / "code" / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(runs).to_csv(ROOT / "code" / "seed_comparison.csv", index=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
