import gymnasium
import numpy as np

class ShapedRewardWrapper(gymnasium.Wrapper):
    """
    A wrapper for HighwayEnv to apply reward shaping during training
    while keeping the original reward intact in info['original_reward']
    for evaluation and comparison purposes.
    """
    def __init__(self, env, lane_change_penalty=0.15, proximity_penalty=0.3, safety_distance=0.25, zigzag_penalty=0.25):
        super().__init__(env)
        self.lane_change_penalty = lane_change_penalty
        self.proximity_penalty = proximity_penalty
        self.safety_distance = safety_distance
        self.zigzag_penalty = zigzag_penalty
        self.current_step = 0
        self.last_lane_change_step = -10

    def reset(self, **kwargs):
        self.current_step = 0
        self.last_lane_change_step = -10
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Save original reward
        info["original_reward"] = reward
        
        # Start with the original reward
        shaped_reward = reward
        
        # 1. Lane change and zigzag penalty (actions: 0 = change lane left, 2 = change lane right)
        if action in [0, 2]:
            shaped_reward -= self.lane_change_penalty
            # Penalize quick consecutive lane changes (within 4 steps)
            if self.current_step - self.last_lane_change_step < 4:
                shaped_reward -= self.zigzag_penalty
            self.last_lane_change_step = self.current_step
            
        # 2. Proximity/Tailgating penalty
        for i in range(1, len(state)):
            presence = state[i][0]
            x = state[i][1]  # positive if in front
            y = state[i][2]  # close to 0 if in the same lane
            
            if presence > 0.5:
                # If vehicle is ahead (x > 0) and in the same/adjacent lane (e.g., abs(y) < 0.15)
                if 0 < x < self.safety_distance and abs(y) < 0.15:
                    # Penalty is higher the closer we are to the vehicle
                    proximity_factor = (self.safety_distance - x) / self.safety_distance
                    shaped_reward -= self.proximity_penalty * proximity_factor
                    
        self.current_step += 1
        return state, shaped_reward, terminated, truncated, info
