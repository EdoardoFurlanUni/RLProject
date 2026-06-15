import gymnasium
import numpy as np

class EnrichedObservationWrapper(gymnasium.ObservationWrapper):
    """
    A wrapper to enrich the standard 5x5 Kinematics observation matrix with:
    1. Ego lane index (normalized)
    2. Time-To-Collision (TTC) to the vehicle directly ahead (normalized)
    3. Left lane occupancy clearance (danger index, where 0.0 is blocked and 1.0 is clear)
    4. Right lane occupancy clearance (danger index, where 0.0 is blocked and 1.0 is clear)
    
    The output is a flattened 29-dimensional vector (25 original features + 4 engineered features).
    """
    def __init__(self, env):
        super().__init__(env)
        # Original state is 5x5 = 25 features. We append 4 engineered features.
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(29,),
            dtype=np.float32
        )

    def observation(self, obs):
        # obs is a 5x5 matrix:
        # Rows: 0 is ego, 1-4 are other vehicles
        # Features: [presence, x, y, vx, vy]
        
        # 1. Ego Lane Index
        ego_y = obs[0][2]  # absolute y position normalized
        try:
            ego_lane = self.env.unwrapped.vehicle.lane_index[2]
            lanes_count = self.env.unwrapped.config.get("lanes_count", 4)
            ego_lane_norm = ego_lane / max(1, lanes_count - 1)
        except Exception:
            # Fallback
            ego_lane_norm = ego_y
            
        # 2. Time-To-Collision (TTC) to front vehicle in the same lane
        ttc = 1.0  # 1.0 means no threat
        min_front_x = 1.0  # normalized longitudinal distance (capped at 100m)
        
        # 3. Adjacent Lane Clearances (Left and Right)
        left_clearance = 1.0   # 1.0 means clear, close to 0.0 means blocked
        right_clearance = 1.0  # 1.0 means clear, close to 0.0 means blocked
        
        # Iterate over surrounding vehicles (1 to 4)
        for i in range(1, len(obs)):
            presence = obs[i][0]
            x = obs[i][1]   # longitudinal relative position
            y = obs[i][2]   # lateral relative position
            vx = obs[i][3]  # relative velocity x
            
            if presence > 0.5:
                # Same lane vehicle (TTC check)
                if x > 0 and abs(y) < 0.1:
                    if x < min_front_x:
                        min_front_x = x
                    if vx < 0:
                        # Relative distance in meters = x * 100
                        # Relative velocity in m/s = vx * 30
                        rel_dist_m = x * 100.0
                        rel_vel_mps = abs(vx * 30.0)
                        current_ttc = rel_dist_m / rel_vel_mps
                        # Normalize TTC: cap at 10 seconds.
                        # Close to 0.0 means immediate collision risk.
                        ttc_norm = min(1.0, current_ttc / 10.0)
                        if ttc_norm < ttc:
                            ttc = ttc_norm
                            
                # Left lane vehicle clearance (-0.3 is typical normalized y for left lane)
                if -0.45 < y < -0.1:
                    distance_factor = abs(x)  # relative x normalized
                    # Cap clearance at 25 meters (0.25 normalized)
                    left_norm = min(1.0, distance_factor / 0.25)
                    if left_norm < left_clearance:
                        left_clearance = left_norm
                        
                # Right lane vehicle clearance (+0.3 is typical normalized y for right lane)
                elif 0.1 < y < 0.45:
                    distance_factor = abs(x)
                    right_norm = min(1.0, distance_factor / 0.25)
                    if right_norm < right_clearance:
                        right_clearance = right_norm
                        
        # Flatten the original 5x5 observation matrix into a 25-dim vector
        flat_obs = obs.flatten()
        
        # Concatenate original features and engineered features
        enriched_obs = np.concatenate([
            flat_obs,
            [ego_lane_norm, ttc, left_clearance, right_clearance]
        ]).astype(np.float32)
        
        return enriched_obs
