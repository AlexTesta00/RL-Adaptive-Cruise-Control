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
model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    gamma=0.95,
    learning_rate=0.00001,
    clip_range=0.2,
    batch_size=128,
    n_steps=2048,
    n_epochs=100,
    ent_coef=0.01,
    seed=1234,
    tensorboard_log="./log")

callback = EvalCallback(env, best_model_save_path='./',
                        log_path='./', eval_freq=1024 * 100,
                        deterministic=True, render=False)

checkpoint_callback = CheckpointCallback(
  save_freq=1024 * 100,
  save_path="./checkpoint/",
  name_prefix="rl_model",
  save_replay_buffer=True
)

# Train the PPO model
model.learn(total_timesteps=1024 * 100, callback=[callback, checkpoint_callback], progress_bar=True, log_interval=1, tb_log_name="ACC - gamma 0.95, rate 0.00001, clip 0.2, batch 128, step 2048, ent_coef 0.01, epoch 100")

# Save the final model
model.save("ppo_adaptive_cruise_control")

# Clean up the environment
env.close()