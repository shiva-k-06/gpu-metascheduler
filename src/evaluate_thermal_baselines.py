import random

from cluster import create_cluster
from workload import generate_jobs
from simulator_thermal import Simulator
from metrics import calculate_metrics


def run_scheduler(name, strategy):
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
    metrics["thermal_violations"] = sim.thermal_violations
    metrics["thermal_violation_steps"] = sim.thermal_violation_steps
    metrics["max_temperature_reached"] = sim.max_temp_seen
    metrics["average_temperature"] = sim.average_temperature()
    metrics["scheduler"] = name

    return metrics


results = [
    run_scheduler("FCFS", "fcfs"),
    run_scheduler("SJF", "sjf"),
    run_scheduler("Round Robin", "round_robin"),
    run_scheduler("Random", "random"),
]

print("\n=== Thermal Baseline Comparison ===")

for result in results:
    print("\nScheduler:", result["scheduler"])
    print("Makespan:", result["makespan"])
    print("Avg Wait Time:", result["avg_wait_time"])
    print("Avg GPU Utilization:", result["avg_gpu_utilization"])
    print("Load Balance Std:", result["load_balance_std"])
    print("Thermal Events:", result["thermal_violations"])
    print("Thermal Violation Steps:", result["thermal_violation_steps"])
    print("Max Temperature:", result["max_temperature_reached"])
    print("Average Temperature:", result["average_temperature"])