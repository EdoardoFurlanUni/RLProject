import os
import json
import matplotlib.pyplot as plt
import numpy as np

# 1. Define the names of the runs and the relative JSON of the learning curve
runs_to_plot = {
    "DQN Training": "results/run_20260404_122349/learning_curve.json"
}

# 2. Moving Average Function (smooths the graph to show the true trend)
def moving_average(a, n=50):
    if len(a) < n:
        n = max(1, len(a) // 2)
        if n == 0:
            return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# 3. Impagina il Plottaggio
plt.figure(figsize=(10, 5))

for name, filepath in runs_to_plot.items():
    if not os.path.exists(filepath):
        print(f"Attenzione! File {filepath} non trovato per '{name}'. Lo ignoro.")
        continue

    print(f"Traccio la learning curve di: {name}")
    with open(filepath, "r") as f:
        data = json.load(f)
        returns = data["training_returns"]

    returns = np.array(returns, dtype=float)
    n_window = min(50, max(1, len(returns)//2))
    smoothed_returns = moving_average(returns, n=n_window)

    line = plt.plot(returns, alpha=0.2)
    color = line[0].get_color()

    if len(smoothed_returns) > 0:
        x_axis = np.arange(n_window - 1, len(returns)) if len(returns) >= n_window else np.arange(len(smoothed_returns))
        plt.plot(x_axis, smoothed_returns, color=color, linewidth=2.5, label=f'{name} ({n_window}-Ep Avg)')


plt.xlabel('Training Episodes')
plt.ylabel('Return')
plt.title('DQN Learning Curves Comparison')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

os.makedirs('plot_data', exist_ok=True)
pdf_path = os.path.join('plot_data', 'learning_curve_plot.pdf')
png_path = os.path.join('plot_data', 'learning_curve_plot.png')

plt.savefig(pdf_path, bbox_inches='tight')
plt.savefig(png_path, bbox_inches='tight', dpi=300)

print(f"\nGrafico salvato con successo in:\n- {pdf_path}")
plt.show()
