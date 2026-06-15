import gymnasium
import highway_env
import numpy as np
import random
import json
import os

def advanced_baseline_policy(state_matrix):

    #sensors to determine better actions
    front_blocked = False
    right_blocked = False
    left_blocked = False
    
    ego_y = state_matrix[0][2] #get vehicle's y position

    if ego_y < 0.1:
        left_blocked = True
    elif ego_y > 0.6:
        right_blocked = True

    for i in range(1,5):
        presence = state_matrix[i][0]
        x = state_matrix[i][1]
        y = state_matrix[i][2]
        vx = state_matrix[i][3]

        if presence == 1.0:
            if 0 < x < 0.1 and abs(y) < 0.1 and vx < 0.0:
                front_blocked = True
            if -0.45 < y < -0.2 and -0.1 < x < 0.1:
                left_blocked = True
            if 0.2 < y < 0.45 and -0.1 < x < 0.1:
                right_blocked = True
    if front_blocked:
        if not right_blocked:
            return 2 #turn right
        elif not left_blocked:
            return 0 #turn left
        else:
            return 4 #slow down
    if not right_blocked:
        return 2

    return 3 #accelerate

env_name = "highway-v0"
env = gymnasium.make(env_name,
                     config={'action': {'type': 'DiscreteMetaAction'}, 'duration': 40, "lanes_count": 3, "ego_spacing": 1.5},
                     render_mode='human')

# Evaluation loop
state, _ = env.reset()
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0
lane_changes = 0
episode_returns = []
episode_lane_changes = []
episode_lengths = []
episode_crashes = []
episode_velocities = []
episode_avg_velocities = []

while episode <= 10:
    episode_steps += 1
    print(state)
    action = advanced_baseline_policy(state)
    if action in [0, 2]:  # 0: Sinistra, 2: Destra
        lane_changes += 1
    state, reward, done, truncated, info = env.step(action)
    episode_velocities.append(info["speed"])
    #state = state.reshape(-1)
    env.render()

    episode_return += reward

    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")
        state, _ = env.reset()
        #state = state.reshape(-1)
        episode += 1

        episode_returns.append(episode_return)
        episode_lengths.append(episode_steps)
        episode_crashes.append(done)  # done = True se si è schiantato, altrimenti False
        episode_lane_changes.append(lane_changes)
        episode_avg_velocities.append(np.mean(episode_velocities))
        episode_steps = 0
        episode_return = 0
        lane_changes = 0
        episode_velocities = []


env.close()
print(f"\n--- Baseline Summary ({episode-1} episodes) ---")
print(f"Average Return: {np.mean(episode_returns):.3f}")
print(f"Average Length: {np.mean(episode_lengths):.3f}")
print(f"Average Crash Rate: {np.mean(episode_crashes):.3f}")
print(f"Average Lane Changes: {np.mean(episode_lane_changes):.3f}")
print(f"Average Speed: {np.mean(episode_avg_velocities):.3f}")

model_name = "advanced_baseline"  
metrics = {
    "returns": episode_returns,
    "lengths": episode_lengths,
    "crashes": [int(c) for c in episode_crashes],
    "lane_changes": episode_lane_changes,
    "avg_velocities": episode_avg_velocities
}
os.makedirs("plot_data", exist_ok=True)
file_path = f"plot_data/{model_name}_metrics.json"
with open(file_path, "w") as f:
    json.dump(metrics, f)
print(f"Metriche salvate con successo in {file_path}!")