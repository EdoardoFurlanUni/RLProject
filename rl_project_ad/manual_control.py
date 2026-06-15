import gymnasium
import highway_env
import json
import os
import numpy as np
# Remember to save what you will need for the plots

env_name = "highway-v0"
env = gymnasium.make(env_name,
                     config={"manual_control": True, "lanes_count": 3, "ego_spacing": 1.5},
                     render_mode='human')

state, _ = env.reset()
done, truncated = False, False
prev_lane = round(state[0][2] / 0.3333)
episode = 1
episode_steps = 0
episode_return = 0
lane_changes = 0
episode_returns = []
episode_lengths = []
episode_crashes = []
episode_lane_changes = []
episode_velocities = []
episode_avg_velocities = []

while episode <= 10:
    episode_steps += 1

    # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
    state, reward, done, truncated, info = env.step(env.action_space.sample())  # With manual control these actions are ignored
    episode_velocities.append(info["speed"])
    curr_lane = round(state[0][2] / 0.3333)
    if curr_lane != prev_lane:
        lane_changes += 1
        prev_lane = curr_lane
        print(f"cambio corsia {lane_changes}")

    env.render()

    episode_return += reward

    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")

        # --- METRICS ---
        episode_returns.append(episode_return)
        episode_lengths.append(episode_steps)
        episode_crashes.append(done)
        episode_lane_changes.append(lane_changes)
        episode_avg_velocities.append(np.mean(episode_velocities))
        state, _ = env.reset()
        prev_lane = round(state[0][2] / 0.3333)
        episode += 1
        episode_steps = 0
        episode_return = 0
        lane_changes = 0
        episode_velocities = []

env.close()

model_name = "manual"  
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
print(f"Metrics saved in {file_path}!")