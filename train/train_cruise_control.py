from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from cruise_control_env import CruiseControlEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


# Trained environment
env = CruiseControlEnv()
env = Monitor(env)

check_env(env)

# Initialize the PPO model
"""
model = PPO(
    "MlpPolicy",  # Use an MLP (feed-forward neural network) policy
    env,
    verbose=0,
    gamma=0.95,
    learning_rate=0.00003,
    batch_size=128,
    n_steps=1024,
    n_epochs=100,
    ent_coef=0.01,
    seed=1234,
    tensorboard_log="./log")
"""

callback = EvalCallback(env, best_model_save_path='./',
                        log_path='./', eval_freq=1000,
                        deterministic=True, render=False)

checkpoint_callback = CheckpointCallback(
  save_freq=1024 * 10,
  save_path="./checkpoint/",
  name_prefix="rl_model",
  save_replay_buffer=True
)

#Get previous checkpoint
model = PPO.load('C:/Users/alext/Desktop/RL-Adaptive-Cruise-Control/model/cruise_control_check_point/rl_model_102400_steps', env=env)

# Train the PPO model
model.learn(total_timesteps=1024 * 500, callback=[callback, checkpoint_callback], progress_bar=True, log_interval=1, tb_log_name="1% Tolerance resume 102400 steps")

# Save the final model
model.save("ppo_carla_model")

# Clean up the environment
env.close()