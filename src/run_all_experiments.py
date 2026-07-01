import random
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from cluster import create_cluster
from workload import generate_jobs
from simulator import Simulator
from metrics import calculate_metrics
from ppo_env import HPCSchedulingEnv


def run_static_scheduler(name, strategy):
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
    return metrics


def run_ppo():
    env = HPCSchedulingEnv()
    model = PPO.load("../models/ppo_gpu_scheduler")

    state, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, truncated, info = env.step(action)

    while len(env.sim.completed_jobs) < len(env.jobs):
        env.sim.step()

    metrics = calculate_metrics(env.sim, len(env.jobs))
    metrics["Scheduler"] = "PPO"
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


results = []

results.append(run_ppo())
results.append(run_static_scheduler("SJF", "sjf"))
results.append(run_static_scheduler("Round Robin", "round_robin"))
results.append(run_static_scheduler("FCFS", "fcfs"))
results.append(run_static_scheduler("Random", "random"))

df = pd.DataFrame(results)

df = df[
    [
        "Scheduler",
        "makespan",
        "avg_wait_time",
        "avg_turnaround_time",
        "avg_gpu_utilization",
        "load_balance_std",
        "completed_jobs",
        "total_jobs"
    ]
]

df.to_csv("../results/all_scheduler_results.csv", index=False)

print("\n=== Final Comparison Results ===")
print(df)

save_chart(df, "makespan", "Makespan Comparison", "Time Units", "makespan_comparison.png")
save_chart(df, "avg_wait_time", "Average Wait Time Comparison", "Time Units", "wait_time_comparison.png")
save_chart(df, "avg_turnaround_time", "Average Turnaround Time Comparison", "Time Units", "turnaround_comparison.png")
save_chart(df, "avg_gpu_utilization", "GPU Utilization Comparison", "Utilization Ratio", "utilization_comparison.png")
save_chart(df, "load_balance_std", "Load Balance Comparison", "Std Dev", "load_balance_comparison.png")

print("\nSaved results and graphs in results folder.")