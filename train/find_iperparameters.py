import optuna
from stable_baselines3 import PPO
from functools import partial
from cruise_control_env import CruiseControlEnv

def evaluate_model(model, env, num_episodes=10):
    all_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_rewards = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            total_rewards += rewards
        all_rewards.append(total_rewards)
    
    mean_rewards = sum(all_rewards) / len(all_rewards)
    return mean_rewards

def optimize_agent(trial):
    # Assumi che env sia definito globalmente o passalo come parametro
    learning_rate = trial.suggest_float('learning_rate', 0.00001, 0.00003)
    gamma = trial.suggest_float('gamma', low=0.95, high=0.98)
    batch_size = trial.suggest_int('batch_size', low=64, high=128)
    
    model = PPO('MlpPolicy', env, verbose=0, gamma=gamma,
                learning_rate=learning_rate, batch_size=batch_size)
    model.learn(total_timesteps=100)
    rewards = evaluate_model(model, env)
    return rewards

env = CruiseControlEnv()  # Assicurati di definire l'ambiente qui

study = optuna.create_study(direction='maximize')
study.optimize(optimize_agent, n_trials=1)

print("Best hyperparameters:", study.best_trial.params)
