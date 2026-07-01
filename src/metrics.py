import numpy as np


def calculate_metrics(sim, total_jobs):

    completed_jobs = sim.completed_jobs

    wait_times = [
        job.start_time - job.arrival_time
        for job in completed_jobs
    ]

    turnaround_times = [
        job.finish_time - job.arrival_time
        for job in completed_jobs
    ]

    makespan = sim.current_time

    gpu_busy_times = []

    for gpu in sim.gpus:
        busy_time = getattr(gpu, "busy_time", 0)
        gpu_busy_times.append(busy_time)

    utilizations = [
        busy_time / makespan if makespan > 0 else 0
        for busy_time in gpu_busy_times
    ]

    return {
        "completed_jobs": len(completed_jobs),
        "total_jobs": total_jobs,
        "makespan": makespan,
        "avg_wait_time": np.mean(wait_times) if wait_times else 0,
        "avg_turnaround_time": np.mean(turnaround_times) if turnaround_times else 0,
        "avg_gpu_utilization": np.mean(utilizations) if utilizations else 0,
        "gpu_utilizations": utilizations,
        "load_balance_std": np.std(utilizations) if utilizations else 0
    }