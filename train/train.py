from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from cruise_control_env import CruiseControlEnv


# Trained environment
env = CruiseControlEnv()

check_env(env)

# Initialize the PPO model
model = PPO(
    "MlpPolicy",  # Use an MLP (feed-forward neural network) policy
    env,
    verbose=1,
    n_steps=2048,
    n_epochs=100,
    tensorboard_log="./log"
    
)
# Train the PPO model
model.learn(total_timesteps=100_000, progress_bar=True)


# Save the final model
model.save("ppo_carla_model")

# Clean up the environment
env.close()