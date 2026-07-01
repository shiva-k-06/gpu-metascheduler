from stable_baselines3 import PPO

from ppo_env_thermal import HPCSchedulingEnv
from metrics import calculate_metrics


env = HPCSchedulingEnv()
model = PPO.load("../models/ppo_gpu_scheduler_thermal")

state, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(state, deterministic=True)
    state, reward, done, truncated, info = env.step(action)

while len(env.sim.completed_jobs) < len(env.jobs):
    env.sim.step()

metrics = calculate_metrics(env.sim, len(env.jobs))

metrics["thermal_violations"] = env.sim.thermal_violations
metrics["thermal_violation_steps"] = env.sim.thermal_violation_steps
metrics["max_temperature_reached"] = env.sim.max_temp_seen
metrics["average_temperature"] = env.sim.average_temperature()
metrics["thermal_violations"] = env.sim.thermal_violations

print("\n=== Thermal PPO Evaluation Metrics ===")
for key, value in metrics.items():
    print(key, ":", value)