import gymnasium
import highway_env
import numpy as np
import torch
import random
from network_bonus import DQNAgent
from reward_wrapper import ShapedRewardWrapper
import os
import datetime
import json
import argparse

# Set seeds
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(1)

def main():
    parser = argparse.ArgumentParser(description="Train a DQN agent with extensions and shaped reward")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory name under results to save models")
    parser.add_argument("--double_dqn", action="store_true", help="Enable Double DQN training")
    parser.add_argument("--dueling", action="store_true", help="Enable Dueling DQN architecture")
    parser.add_argument("--steps", type=int, default=20000, help="Total training steps")
    parser.add_argument("--lane_change_penalty", type=float, default=0.15, help="Reward penalty for lane changes")
    parser.add_argument("--proximity_penalty", type=float, default=0.30, help="Reward penalty for tailgating/proximity")
    parser.add_argument("--safety_distance", type=float, default=0.25, help="Longitudinal distance to start penalizing proximity")
    parser.add_argument("--state_mod", action="store_true", help="Enable enriched state representation (29 features)")
    parser.add_argument("--lanes_count", type=int, default=4, help="Number of lanes in the highway environment")
    parser.add_argument("--vehicles_count", type=int, default=50, help="Number of vehicles in the environment")
    parser.add_argument("--high_speed_reward", type=float, default=0.4, help="Reward weight for high speed")
    parser.add_argument("--right_lane_reward", type=float, default=0.1, help="Reward weight for right lane in env config")
    parser.add_argument("--env_lane_change_reward", type=float, default=-0.1, help="Lane change reward in env config")
    parser.add_argument("--zigzag_penalty", type=float, default=0.25, help="Reward penalty for quick consecutive lane changes")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--buffer_size", type=int, default=15000, help="Replay buffer capacity")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient tau")
    parser.add_argument("--epsilon_decay", type=int, default=5000, help="Steps for epsilon to decay to epsilon_end")
    args = parser.parse_args()

    # Create directory for saving results
    save_dir = os.path.join(os.path.dirname(__file__), "results", args.model_dir)
    os.makedirs(save_dir, exist_ok=True)

    env_name = "highway-fast-v0"
    
    # Create target environment with user's configurations
    base_env = gymnasium.make(env_name,
                             config={
                                 'action': {'type': 'DiscreteMetaAction'},
                                 'duration': 40,
                                 'lanes_count': args.lanes_count,
                                 'vehicles_count': args.vehicles_count,
                                 'lane_change_reward': args.env_lane_change_reward,
                                 'right_lane_reward': args.right_lane_reward,
                                 'high_speed_reward': args.high_speed_reward,
                                 'reward_speed_range': [25, 30]
                             })
    
    # Wrap with custom reward shaping wrapper
    env = ShapedRewardWrapper(
        base_env,
        lane_change_penalty=args.lane_change_penalty,
        proximity_penalty=args.proximity_penalty,
        safety_distance=args.safety_distance,
        zigzag_penalty=args.zigzag_penalty
    )

    # If state modification is enabled, wrap observation and update state_dim
    state_dim = 25
    if args.state_mod:
        from observation_wrapper import EnrichedObservationWrapper
        env = EnrichedObservationWrapper(env)
        state_dim = 29

    # Initialize DQN Agent with specified extensions
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=5,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        tau=args.tau,
        epsilon_decay=args.epsilon_decay,
        double_dqn=args.double_dqn,
        dueling=args.dueling
    )

    state, _ = env.reset()
    state = state.reshape(-1)
    
    episode = 1
    episode_steps = 0
    episode_return = 0        # accumulates original reward
    episode_shaped_return = 0 # accumulates shaped reward
    
    episode_returns = []
    episode_shaped_returns = []
    episode_crashes = []

    print(f"Starting training for {args.model_dir} (Double DQN: {args.double_dqn}, Dueling: {args.dueling})...")
    print(f"Reward shaping params: LC Penalty={args.lane_change_penalty}, Prox Penalty={args.proximity_penalty}, Safety Dist={args.safety_distance}, Zigzag Penalty={args.zigzag_penalty}")
    print(f"Env config params: Right Lane Reward={args.right_lane_reward}, Env LC Reward={args.env_lane_change_reward}, High Speed Reward={args.high_speed_reward}")

    try:
        for t in range(args.steps):
            episode_steps += 1
            action = agent.select_action(state)

            next_state, shaped_reward, done, truncated, info = env.step(action)
            next_state = next_state.reshape(-1)

            # Store shaped transition in memory and train at every step
            agent.store_transition(state, action, shaped_reward, next_state, done)
            agent.train_step()
            
            state = next_state
            
            # Track both rewards
            original_reward = info.get("original_reward", shaped_reward)
            episode_return += original_reward
            episode_shaped_return += shaped_reward

            # Periodic saving
            if t > 0 and t % 2000 == 0:
                interim_path = os.path.join(save_dir, f"dqn_model_step_{t}.pth")
                agent.save(interim_path)
                print(f"[{t}/{args.steps}] Interim model saved.")

            if done or truncated:
                print(f"Total T: {t} | Episode: {episode} | Steps: {episode_steps} | Return (Orig): {episode_return:.3f} | Return (Shaped): {episode_shaped_return:.3f} | Epsilon: {agent.epsilon():.3f}")

                episode_returns.append(episode_return)
                episode_shaped_returns.append(episode_shaped_return)
                episode_crashes.append(done)

                state, _ = env.reset()
                state = state.reshape(-1)
                
                episode += 1
                episode_steps = 0
                episode_return = 0
                episode_shaped_return = 0
                
    except KeyboardInterrupt:
        print("Training interrupted.")

    env.close()

    # Save final model
    model_path = os.path.join(save_dir, "dqn_model.pth")
    agent.save(model_path)
    
    # Save training curves
    learning_data = {
        "training_returns": episode_returns,
        "training_shaped_returns": episode_shaped_returns
    }
    learning_path = os.path.join(save_dir, "learning_curve.json")
    with open(learning_path, "w") as f:
        json.dump(learning_data, f)
        
    print(f"Training complete! Final model saved to {model_path}")
    print(f"Learning curves saved to {learning_path}")
    if len(episode_returns) > 0:
        print(f"Avg original return (last 20 ep): {np.mean(episode_returns[-20:]):.3f}")
        print(f"Crash rate (last 20 ep): {np.mean(episode_crashes[-20:]):.1%}")

if __name__ == "__main__":
    main()
