import gymnasium
import highway_env
import json
import os

# Remember to save what you will need for the plots

env_name = "highway-v0"
env = gymnasium.make(env_name,
                     config={"manual_control": True, "lanes_count": 3, "ego_spacing": 1.5},
                     render_mode='human')

env.reset()
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0
episode_returns = []
episode_lengths = []
episode_crashes = []

while episode <= 10:
    episode_steps += 1

    # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
    _, reward, done, truncated, _ = env.step(env.action_space.sample())  # With manual control these actions are ignored
    env.render()

    episode_return += reward

    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")

        # --- RACCOLTA METRICHE ---
        episode_returns.append(episode_return)
        episode_lengths.append(episode_steps)
        episode_crashes.append(done)  # done = True se si è schiantato, altrimenti False

        env.reset()
        episode += 1
        episode_steps = 0
        episode_return = 0

env.close()

model_name = "manual"  
metrics = {
    "returns": episode_returns,
    "lengths": episode_lengths,
    "crashes": [int(c) for c in episode_crashes] 
}
os.makedirs("plot_data", exist_ok=True)
file_path = f"plot_data/{model_name}_metrics.json"
with open(file_path, "w") as f:
    json.dump(metrics, f)
print(f"Metriche salvate con successo in {file_path}!")