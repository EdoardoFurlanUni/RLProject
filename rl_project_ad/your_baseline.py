import gymnasium
import highway_env
import numpy as np
import random
import json

# standard baseline policy, not used in the results
def baseline_policy(state_matrix):

    action = 3 #if no other event occur, accelerate
    for i in range(1,5):
        presence = state_matrix[i][0]
        x = state_matrix[i][1]
        y = state_matrix[i][2]
        vx = state_matrix[i][3]
        vy = state_matrix[i][4]
        if presence == 1.0:
            if 0 < x < 0.2 and abs(y) < 0.15:
                return 0 #turn left
    return action
env_name = "highway-v0"
env = gymnasium.make(env_name,
                     config={'action': {'type': 'DiscreteMetaAction'}, 'duration': 40, "lanes_count": 3},
                     render_mode='human')

# Evaluation loop
state, _ = env.reset()
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0
episode_returns = []

while episode <= 10:
    episode_steps += 1
    print(state)
    action = baseline_policy(state)
    state, reward, done, truncated, _ = env.step(action)
    #state = state.reshape(-1)
    env.render()

    episode_return += reward

    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")
        episode_returns.append(episode_return)
        state, _ = env.reset()
        #state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

env.close()
print(f"\n--- Baseline Summary ({episode-1} episodes) ---")
print(f"Average Return: {np.mean(episode_returns):.3f}")