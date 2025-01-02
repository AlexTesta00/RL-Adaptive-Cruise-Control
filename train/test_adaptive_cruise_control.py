import sys, os
from stable_baselines3 import PPO
from adaptive_cruise_control_env import AdaptiveCruiseControlEnv
import gymnasium


# Trained environment
env = AdaptiveCruiseControlEnv()

# Save the final model
env.set_target_speed(10)
trainedModel = PPO.load("C:/Users/alext/Desktop/RL-Adaptive-Cruise-Control/best_model", env=env)
# trainedModel.get_env()
obs = trainedModel.get_env().reset()
# obs[0][2] = 110.0
print(obs)

while True:
    action, _states = trainedModel.predict(obs, deterministic=True)
    obs, rewards, dones, info = trainedModel.get_env().step(action)


# Clean up the environment
# env.close()