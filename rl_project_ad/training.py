import gymnasium
import highway_env
import numpy as np
import torch
import random
from network import DQNAgent
import os
import datetime
import json

# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(os.path.dirname(__file__), "results", f"run_{timestamp}")
os.makedirs(save_dir, exist_ok=True)

MAX_STEPS = int(2e4)  # This should be enough to obtain nice results, however feel free to change it
env_name = "highway-fast-v0"  # We use the 'fast' env just for faster training, if you want you can use "highway-v0"

env = gymnasium.make(env_name,
                     config={'action': {'type': 'DiscreteMetaAction'}, 'duration': 40, 'vehicles_count':50, 'lane_change_reward': -0.3, 'right_lane_reward': 0.4, 'reward_speed_range': [25,30]})

# Initialize your model
agent = DQNAgent(state_dim=25, action_dim=5)


state, _ = env.reset()
state = state.reshape(-1)
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0
episode_returns = []
episode_crashes = []
# Training loop
try: #add try-except block to save the model's learning curve even when the training is intrerupted
    for t in range(MAX_STEPS):
        episode_steps += 1
        # Select the action to be performed by the agent
        action = agent.select_action(state)

        # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = next_state.reshape(-1)

        # Store transition in memory and train your model
        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step()
        state = next_state
        episode_return += reward

        if t > 0 and t % 2000 == 0:
            interim_model_path = os.path.join(save_dir, f"dqn_model_step_{t}.pth")
            agent.save(interim_model_path)
            print(f"[{t}/{MAX_STEPS}] Salvataggio intermedio completato!")

        if done or truncated:
            print(f"Total T: {t} Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}")

            # Save training information
            episode_returns.append(episode_return)
            episode_crashes.append(done)

            state, _ = env.reset()
            state = state.reshape(-1)
            episode += 1
            episode_steps = 0
            episode_return = 0
except KeyboardInterrupt:
    print("Training interrupted by user.")

env.close()

# Salva l'agente nella nuova cartella
model_path = os.path.join(save_dir, "dqn_model.pth")
agent.save(model_path)
learning_data = {
    "training_returns": episode_returns
}
learning_path = os.path.join(save_dir, "learning_curve.json")
with open(learning_path, "w") as f:
    json.dump(learning_data, f)
print(f"Learning curve saved in {learning_path}!")
print(f"\nTraining complete! {episode - 1} episodes over {MAX_STEPS} steps.")
print(f"Avg return (last 20 ep): {np.mean(episode_returns[-20:]):.3f}")
print(f"Crash rate (last 20 ep): {np.mean(episode_crashes[-20:]):.1%}")
print(f"--> Results saved in: {save_dir}")