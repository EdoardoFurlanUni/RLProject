import gymnasium
import highway_env
import numpy as np
import random

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
    action = advanced_baseline_policy(state)
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