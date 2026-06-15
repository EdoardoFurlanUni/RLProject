import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Get dynamic path to the plot_data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
plot_data_dir = os.path.join(script_dir, "plot_data")

groups = {
    "p13": {
        "Manual Human": os.path.join(plot_data_dir, "manual_metrics.json"),
        "Base DQN": os.path.join(plot_data_dir, "base_dqn_metrics.json"),
        "Advanced Baseline": os.path.join(plot_data_dir, "advanced_baseline_metrics.json"),
        "P1.1: Shape (Weak)": os.path.join(plot_data_dir, "phase1_reward_shaping_metrics.json"),
        "P1.2: Shape (Strict)": os.path.join(plot_data_dir, "phase1_reward_shaping_strict_metrics.json"),
        "P2: Env Config": os.path.join(plot_data_dir, "phase2_env_config_metrics.json"),
        "P3.1: State (Short Exp)": os.path.join(plot_data_dir, "phase3_state_representation_metrics.json"),
        "P3.2: State (Long Exp)": os.path.join(plot_data_dir, "phase3_state_representation_long_metrics.json")
    },
    "p4": {
        "Manual Human": os.path.join(plot_data_dir, "manual_metrics.json"),
        "Base DQN": os.path.join(plot_data_dir, "base_dqn_metrics.json"),
        "Advanced Baseline": os.path.join(plot_data_dir, "advanced_baseline_metrics.json"),
        "P4: DDQN (12k)": os.path.join(plot_data_dir, "phase4_double_dqn_metrics.json"),
        "P4: DDQN (3 Lanes)": os.path.join(plot_data_dir, "phase4_double_dqn_3lanes_metrics.json"),
        "P4: DDQN (Balanced)": os.path.join(plot_data_dir, "phase4_double_dqn_overtake_balanced_metrics.json"),
        "P4: DDQN (Tuned)": os.path.join(plot_data_dir, "phase4_double_dqn_overtake_courageous_metrics.json")
    }
}

for group_name, models in groups.items():
    results = {}
    for name, filepath in models.items():
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
                results[name] = {
                    "returns_mean": np.mean(data["returns"]),
                    "returns_std": np.std(data["returns"]),
                    "lengths_mean": np.mean(data["lengths"]),
                    "lengths_std": np.std(data["lengths"]),
                    "crash_rate": np.mean(data["crashes"]) * 100,
                    "lane_changes_mean": np.mean(data["lane_changes"]),
                    "lane_changes_std": np.std(data["lane_changes"]),
                    "velocities_mean": np.mean(data.get("avg_velocities", [27.02] * len(data["returns"]))),
                    "velocities_std": np.std(data.get("avg_velocities", [27.02] * len(data["returns"])))
                }
        else:
            print(f"Nota: file non trovato per {name} ({os.path.basename(filepath)}) nel gruppo {group_name}. Verrà escluso.")

    if not results:
        print(f"Nessun dato per il gruppo {group_name}!")
        continue

    labels = list(results.keys())
    returns = [results[m]["returns_mean"] for m in labels]
    returns_std = [results[m]["returns_std"] for m in labels]
    lengths = [results[m]["lengths_mean"] for m in labels]
    lengths_std = [results[m]["lengths_std"] for m in labels]
    crashes = [results[m]["crash_rate"] for m in labels]
    lane_changes = [results[m]["lane_changes_mean"] for m in labels]
    lane_changes_std = [results[m]["lane_changes_std"] for m in labels]
    velocities = [results[m]["velocities_mean"] for m in labels]
    velocities_std = [results[m]["velocities_std"] for m in labels]

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3', '#8C8C8C', '#CCB974'][:len(labels)]

    # --- PLOT 1: RETURNS ---
    plt.figure(figsize=(8, 4))
    plt.bar(labels, returns, yerr=returns_std, capsize=5, color=colors, alpha=0.8)
    plt.ylabel('Average Return', fontsize=10)
    plt.title(f'Comparison of Agent Returns ({group_name.upper()})', fontsize=12, fontweight='bold')
    plt.xticks(rotation=25, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_data_dir, f'comparison_returns_{group_name}.pdf'))
    plt.close()

    # --- PLOT 2: SURVIVAL ---
    plt.figure(figsize=(8, 4))
    plt.bar(labels, lengths, yerr=lengths_std, capsize=5, color=colors, alpha=0.8)
    plt.ylabel('Average Steps (Max 40)', fontsize=10)
    plt.title(f'Comparison of Survival Time ({group_name.upper()})', fontsize=12, fontweight='bold')
    plt.xticks(rotation=25, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_data_dir, f'comparison_lengths_{group_name}.pdf'))
    plt.close()

    # --- PLOT 3: CRASH RATE ---
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, crashes, color=colors, alpha=0.8)
    plt.ylabel('Crash Rate (%)', fontsize=10)
    plt.title(f'Crash Rate Percentage ({group_name.upper()})', fontsize=12, fontweight='bold')
    plt.ylim(0, 100)
    plt.xticks(rotation=25, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', va='bottom', ha='center', fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_data_dir, f'comparison_crashes_{group_name}.pdf'))
    plt.close()

    # --- PLOT 4: LANE CHANGES ---
    plt.figure(figsize=(8, 4))
    plt.bar(labels, lane_changes, yerr=lane_changes_std, capsize=5, color=colors, alpha=0.8)
    plt.ylabel('Average Lane Changes', fontsize=10)
    plt.title(f'Agent Stability: Lane Changes ({group_name.upper()})', fontsize=12, fontweight='bold')
    plt.xticks(rotation=25, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_data_dir, f'comparison_lane_changes_{group_name}.pdf'))
    plt.close()

    # --- PLOT 5: VELOCITY ---
    plt.figure(figsize=(8, 4))
    plt.bar(labels, velocities, yerr=velocities_std, capsize=5, color=colors, alpha=0.8)
    plt.ylabel('Average Velocity (m/s)', fontsize=10)
    plt.title(f'Velocity Comparison ({group_name.upper()})', fontsize=12, fontweight='bold')
    plt.xticks(rotation=25, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_data_dir, f'comparison_velocities_{group_name}.pdf'))
    plt.close()

print("Grafici generati con successo per entrambi i gruppi!")