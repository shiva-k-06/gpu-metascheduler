import random
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from cluster import create_cluster
from workload import generate_jobs
from simulator_thermal import Simulator
from metrics import calculate_metrics
from ppo_env_thermal import HPCSchedulingEnv


def run_thermal_ppo():
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
    metrics["Scheduler"] = "PPO Thermal"
    metrics["thermal_violations"] = env.sim.thermal_violations
    metrics["thermal_violation_steps"] = env.sim.thermal_violation_steps
    metrics["max_temperature_reached"] = env.sim.max_temp_seen
    metrics["average_temperature"] = env.sim.average_temperature()

    return metrics


def run_baseline(name, strategy):
    gpus = create_cluster()
    jobs = generate_jobs()
    sim = Simulator(gpus, jobs)

    if strategy == "random":
        for job in jobs:
            gpu_id = random.randint(0, len(gpus) - 1)
            sim.assign_job_to_gpu(job, gpu_id)

    elif strategy == "round_robin":
        gpu_index = 0
        for job in jobs:
            sim.assign_job_to_gpu(job, gpu_index)
            gpu_index = (gpu_index + 1) % len(gpus)

    elif strategy == "fcfs":
        for job in jobs:
            best_gpu = min(gpus, key=lambda gpu: len(gpu.queue))
            sim.assign_job_to_gpu(job, best_gpu.gpu_id)

    elif strategy == "sjf":
        jobs = sorted(jobs, key=lambda job: job.runtime)
        for job in jobs:
            best_gpu = min(gpus, key=lambda gpu: len(gpu.queue))
            sim.assign_job_to_gpu(job, best_gpu.gpu_id)

    while len(sim.completed_jobs) < len(jobs):
        sim.step()

    metrics = calculate_metrics(sim, len(jobs))
    metrics["Scheduler"] = name
    metrics["thermal_violations"] = sim.thermal_violations
    metrics["thermal_violation_steps"] = sim.thermal_violation_steps
    metrics["max_temperature_reached"] = sim.max_temp_seen
    metrics["average_temperature"] = sim.average_temperature()

    return metrics


def save_chart(df, column, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.bar(df["Scheduler"], df[column])
    plt.title(title)
    plt.xlabel("Scheduler")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"../results/{filename}")
    plt.close()


results = [
    run_thermal_ppo(),
    run_baseline("SJF", "sjf"),
    run_baseline("Round Robin", "round_robin"),
    run_baseline("FCFS", "fcfs"),
    run_baseline("Random", "random"),
]

df = pd.DataFrame(results)

df = df[
    [
        "Scheduler",
        "makespan",
        "avg_wait_time",
        "avg_turnaround_time",
        "avg_gpu_utilization",
        "load_balance_std",
        "thermal_violations",
        "thermal_violation_steps",
        "max_temperature_reached",
        "average_temperature",
        "completed_jobs",
        "total_jobs",
    ]
]

df.to_csv("../results/thermal_scheduler_results.csv", index=False)

print("\n=== Thermal Scheduler Comparison ===")
print(df)

save_chart(
    df,
    "thermal_violation_steps",
    "Thermal Violation Steps Comparison",
    "Steps",
    "thermal_violation_steps.png",
)

save_chart(
    df,
    "average_temperature",
    "Average GPU Temperature Comparison",
    "Temperature (°C)",
    "average_temperature.png",
)

save_chart(
    df,
    "max_temperature_reached",
    "Maximum GPU Temperature Comparison",
    "Temperature (°C)",
    "max_temperature.png",
)

print("\nThermal CSV and graphs saved in results folder.")