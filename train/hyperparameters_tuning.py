import optuna
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from adaptive_cruise_control_env import AdaptiveCruiseControlEnv
from stable_baselines3.common.monitor import Monitor


def evaluate_policy(model, env, n_eval_episodes=1):
    """
    Evaluate a RL agent
    :param model: (BaseAlgorithm) The RL Agent
    :param env: (Gym Environment) The environment
    :param n_eval_episodes: (int) Number of episodes to evaluate the agent
    :return: (float, float) Mean reward and standard deviation of reward
    """
    all_episode_rewards = []
    for _ in range(n_eval_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_rewards.append(reward)
        all_episode_rewards.append(sum(episode_rewards))

    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    return mean_reward, std_reward

def optimize_ppo(trial):
    # Define the hyperparameter search space
    gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    clip_range = trial.suggest_float('clip_range', low=0.1, high=0.3, step=0.1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096])
    ent_coef = trial.suggest_float('ent_coef', low=0.00001, high=0.01, log=True)

    # Trained environment
    env = AdaptiveCruiseControlEnv()
    env = Monitor(env)
    check_env(env)

    # Initialize the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        gamma=gamma,
        learning_rate=learning_rate,
        clip_range=clip_range,
        batch_size=batch_size,
        n_steps=n_steps,
        n_epochs=1,
        ent_coef=ent_coef,
        seed=1234,
    )

    # Train the PPO model
    model.learn(total_timesteps=1024 * 30, progress_bar=True, log_interval=1)

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1)

    # Clean up the environment
    env.close()

    return mean_reward

# Create an Optuna study and optimize the objective function
study = optuna.create_study(
    direction='maximize',
    storage='sqlite:///db.sqlite3',
    study_name='ppo_hyperparameter_tuning'
    )

study.optimize(optimize_ppo, n_trials=1)

# Print the best hyperparameters
print(f"Best hyperparameters: {study.best_params}, Best value: {study.best_value}")