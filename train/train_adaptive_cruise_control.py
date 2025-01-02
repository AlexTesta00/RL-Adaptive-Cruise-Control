from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from adaptive_cruise_control_env import AdaptiveCruiseControlEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


# Trained environment
env = AdaptiveCruiseControlEnv()
env = Monitor(env)

check_env(env)

# Initialize the PPO model

# model = PPO(
#     "MlpPolicy",  # Use an MLP (feed-forward neural network) policy
#     env,
#     verbose=0,
#     gamma=0.95,
#     learning_rate=0.00003,
#     batch_size=128,
#     n_steps=1024,
#     n_epochs=100,
#     ent_coef=0.01,
#     seed=1234,
#     tensorboard_log="./log")

callback = EvalCallback(env, best_model_save_path='./',
                        log_path='./', eval_freq=1024 * 50,
                        deterministic=True, render=False)

checkpoint_callback = CheckpointCallback(
  save_freq=1024 * 50,
  save_path="./checkpoint/",
  name_prefix="rl_model",
  save_replay_buffer=True
)

model = PPO.load("C:/Users/alext/Desktop/RL-Adaptive-Cruise-Control/best_model", env=env)

# Train the PPO model
model.learn(total_timesteps=1024 * 500, callback=[callback, checkpoint_callback], progress_bar=True, log_interval=1, tb_log_name="Adaptive Cruise Control Training")

# Save the final model
model.save("ppo_adaptive_cruise_control")

# Clean up the environment
env.close()