import gymnasium
import highway_env
env = gymnasium.make("highway-v0")
print(env.unwrapped.config)