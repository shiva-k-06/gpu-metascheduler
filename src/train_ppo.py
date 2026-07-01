from stable_baselines3 import PPO
from ppo_env import HPCSchedulingEnv


env = HPCSchedulingEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99
)

model.learn(total_timesteps=100000)

model.save("../models/ppo_gpu_scheduler")

print("Training finished and model saved.")