import sys, os
from stable_baselines3 import PPO
from adaptive_cruise_control_env import AdaptiveCruiseControlEnv
import gymnasium


# Trained environment
# env = AccEnv()
env = AdaptiveCruiseControlEnv()

# env.reset()
# env = gymnasium.wrappers.atari_preprocessing.AtariPreprocessing(env, frame_skip = 4)

# env = VecMonitor(SubprocVecEnv([AccEnv() for i in range(10)]))

# # Initialize the PPO model
# model = PPO(
#     "MlpPolicy",  # Use an MLP (feed-forward neural network) policy
#     env,
#     verbose=0,
#     gamma=0.99,  # Discount factor
#     learning_rate=0.0003,  # Learning rate
#     n_steps=1024,  # Rollout buffer size
#     batch_size=64,  # Batch size for training
#     ent_coef=0.01,  # Entropy coefficient
#     n_epochs=500,
#     seed=42,
#     tensorboard_log="./log"
    
# )
# # env = VecMonitor(SubprocVecEnv([AccEnv() for i in range(10)]), filename=None)
# # Train the PPO model
# model.learn(total_timesteps=1024*500, progress_bar=True, log_interval=1)


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