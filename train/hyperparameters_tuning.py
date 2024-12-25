import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from adaptive_cruise_control_env import AdaptiveCruiseControlEnv


def optimize_ppo(trial):

    env = make_vec_env(lambda: AdaptiveCruiseControlEnv(), n_envs=1, seed=1234)

    gamma = trial.suggest_categorical('gamma', [0.99, 0.98, 0.97, 0.96, 0.95])
    learning_rate = trial.suggest_categorical('learning_rate', [0.00003, 0.0003, 0.003])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    normalize_advantage = trial.suggest_categorical('normalize_advantage', [True, False])

    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=0, 
        tensorboard_log=None,
        gamma=gamma, 
        learning_rate=learning_rate, 
        n_steps=2048, 
        ent_coef=0.01,
        batch_size=batch_size,
        clip_range=0.2,
        normalize_advantage=normalize_advantage,
        n_epochs=30)
    

    model.learn(total_timesteps=2048 * 30, progress_bar=True)
    cumulated_reward = 0
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, _ = env.step(action)
        cumulated_reward += rewards
        if done:
            break
    
    env.close()
    return cumulated_reward

study = optuna.create_study(
        direction='maximize',
        storage="sqlite:///db.sqlite3",
        study_name="adaptive-cruise-control-ppo"
    )
study.optimize(optimize_ppo, n_trials=1)

print("Best:", study.best_params)