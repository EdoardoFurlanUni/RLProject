import json
import os
import numpy as np
import matplotlib.pyplot as plt

models_to_compare = {
    "Manual Human": "plot_data/manual_metrics.json" 
}

results = {}

# 1. Carica i Dati
for name, filepath in models_to_compare.items():
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            # Calcola le medie e l'errore per plottare in Bar Chart
            results[name] = {
                "returns_mean": np.mean(data["returns"]),
                "returns_std": np.std(data["returns"]),
                "lengths_mean": np.mean(data["lengths"]),
                "lengths_std": np.std(data["lengths"]),
                # Crash rate: media di [0,1,0,0,1...] = % di crash!
                "crash_rate": np.mean(data["crashes"]) * 100 
            }
    else:
        print(f"Attenzione: file non trovato per {name}! Avvia lo script prima per generare i dati.")

if not results:
    print("Nessun dato trovato! Non c'è nulla da plottare.")
    exit()

labels = list(results.keys())
returns = [results[m]["returns_mean"] for m in labels]
returns_std = [results[m]["returns_std"] for m in labels]

lengths = [results[m]["lengths_mean"] for m in labels]
lengths_std = [results[m]["lengths_std"] for m in labels]

crashes = [results[m]["crash_rate"] for m in labels]

# --- PLOT 1: AVERAGE RETURNS (Bar Chart with Error Bars) ---
plt.figure(figsize=(8, 5))
plt.bar(labels, returns, yerr=returns_std, capsize=5, color=['#4C72B0', '#55A868', '#C44E52'], alpha=0.8)
plt.ylabel('Average Return per Episode')
plt.title('Comparison of Agent Returns')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('plot_data/comparison_returns.pdf')  # LaTeX ama il pdf
plt.show()

# --- PLOT 2: EPISODE LENGTHS (Durata) ---
plt.figure(figsize=(8, 5))
plt.bar(labels, lengths, yerr=lengths_std, capsize=5, color=['#4C72B0', '#55A868', '#C44E52'], alpha=0.8)
plt.ylabel('Average Steps per Episode (Max 40)')
plt.title('Comparison of Agent Survival Time')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('plot_data/comparison_lengths.pdf')
plt.show()

# --- PLOT 3: CRASH RATE ---
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, crashes, color=['#CC3333', '#CC3333', '#CC3333'], alpha=0.8)
plt.ylabel('Crash Rate (%)')
plt.title('Safety Comparison: Crash Rate Percentage')
plt.ylim(0, 100)
# Aggiungi il numero scritto sopra la colonnina del crash rate
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', va='bottom', ha='center')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('plot_data/comparison_crashes.pdf')
plt.show()
