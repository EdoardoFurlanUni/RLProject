import gymnasium
import highway_env
import numpy as np
import torch
import random
import os
import json
import argparse
from network_bonus import DQNAgent

# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent on HighwayEnv")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory name under results containing dqn_model.pth")
    parser.add_argument("--model_name", type=str, required=True, help="Model name prefix for saving metrics (e.g., base_dqn, double_dqn)")
    parser.add_argument("--dueling", action="store_true", help="Use Dueling DQN architecture")
    parser.add_argument("--double_dqn", action="store_true", help="Use Double DQN")
    parser.add_argument("--state_mod", action="store_true", help="Use enriched state representation (29 features)")
    parser.add_argument("--render", action="store_true", help="Render the environment (human mode)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--model_file", type=str, default="dqn_model.pth", help="Specific checkpoint filename to load")
    parser.add_argument("--lanes_count", type=int, default=3, help="Number of lanes in the highway environment")
    parser.add_argument("--vehicles_count", type=int, default=50, help="Number of vehicles in the environment")
    args = parser.parse_args()

    env_name = "highway-v0"
    render_mode = "human" if args.render else None

    env = gymnasium.make(env_name,
                         config={
                             'action': {'type': 'DiscreteMetaAction'}, 
                             "lanes_count": args.lanes_count, 
                             "vehicles_count": args.vehicles_count,
                             "ego_spacing": 1.5
                         },
                         render_mode=render_mode)

    # Wrap with EnrichedObservationWrapper if state_mod is enabled
    state_dim = 25
    if args.state_mod:
        from observation_wrapper import EnrichedObservationWrapper
        env = EnrichedObservationWrapper(env)
        state_dim = 29

    # Initialize agent with specified architecture and load weights
    agent = DQNAgent(state_dim=state_dim, action_dim=5, dueling=args.dueling, double_dqn=args.double_dqn)
    model_path = os.path.join(os.path.dirname(__file__), "results", args.model_dir, args.model_file)
    agent.load(model_path)

    # Evaluation loop
    state, _ = env.reset()
    state = state.reshape(-1)
    done, truncated = False, False

    episode = 1
    episode_steps = 0
    episode_return = 0
    lane_changes = 0
    
    episode_lane_changes = []
    episode_returns = []
    episode_lengths = []
    episode_crashes = []
    episode_velocities = []
    episode_avg_velocities = []

    while episode <= args.episodes:
        episode_steps += 1
        # Select the action to be performed by the agent
        action = agent.select_action(state, evaluate=True)
        if action in [0, 2]:  # 0: Sinistra, 2: Destra
            lane_changes += 1
        state, reward, done, truncated, info = env.step(action)
        episode_velocities.append(info["speed"])
        state = state.reshape(-1)
        if args.render:
            env.render()

        episode_return += reward

        if done or truncated:
            print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")
            # --- RACCOLTA METRICHE ---
            episode_returns.append(episode_return)
            episode_lengths.append(episode_steps)
            episode_crashes.append(done)  # done = True se si è schiantato, altrimenti False
            episode_lane_changes.append(lane_changes)
            episode_avg_velocities.append(np.mean(episode_velocities))
            
            state, _ = env.reset()
            state = state.reshape(-1)
            episode += 1
            episode_steps = 0
            episode_return = 0
            lane_changes = 0
            episode_velocities = []

    env.close()
    
    metrics = {
        "returns": episode_returns,
        "lengths": episode_lengths,
        "crashes": [int(c) for c in episode_crashes],
        "lane_changes": episode_lane_changes,
        "avg_velocities": episode_avg_velocities
    }
    plot_data_dir = os.path.join(os.path.dirname(__file__), "plot_data")
    os.makedirs(plot_data_dir, exist_ok=True)
    file_path = os.path.join(plot_data_dir, f"{args.model_name}_metrics.json")
    with open(file_path, "w") as f:
        json.dump(metrics, f)
    print(f"Metriche salvate con successo in {file_path}!")

if __name__ == "__main__":
    main()
