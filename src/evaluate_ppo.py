from stable_baselines3 import PPO

from ppo_env import HPCSchedulingEnv
from metrics import calculate_metrics


env = HPCSchedulingEnv()
model = PPO.load("../models/ppo_gpu_scheduler")

state, _ = env.reset()

done = False

while not done:
    action, _ = model.predict(state, deterministic=True)
    state, reward, done, truncated, info = env.step(action)

# Let all queued/running jobs finish
while len(env.sim.completed_jobs) < len(env.jobs):
    env.sim.step()

metrics = calculate_metrics(env.sim, len(env.jobs))

print("\n=== PPO Evaluation Metrics ===")
for key, value in metrics.items():
    print(key, ":", value)