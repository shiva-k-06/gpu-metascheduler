import pandas as pd


results = [
    {
        "Scheduler": "PPO",
        "Makespan": 1316,
        "Avg Wait Time": 530.8,
        "Avg Turnaround Time": 564.88,
        "Avg GPU Utilization": 0.8886,
        "Load Balance Std": 0.0843
    },
    {
        "Scheduler": "SJF",
        "Makespan": 1746,
        "Avg Wait Time": 396.73,
        "Avg Turnaround Time": 434.53,
        "Avg GPU Utilization": 0.7407,
        "Load Balance Std": 0.2004
    },
    {
        "Scheduler": "Round Robin",
        "Makespan": 1640,
        "Avg Wait Time": 572.5,
        "Avg Turnaround Time": 610.63,
        "Avg GPU Utilization": 0.7953,
        "Load Balance Std": 0.1859
    },
    {
        "Scheduler": "FCFS",
        "Makespan": 2070,
        "Avg Wait Time": 636.4,
        "Avg Turnaround Time": 678.68,
        "Avg GPU Utilization": 0.6969,
        "Load Balance Std": 0.2215
    },
    {
        "Scheduler": "Random",
        "Makespan": 2092,
        "Avg Wait Time": 698.34,
        "Avg Turnaround Time": 743.3,
        "Avg GPU Utilization": 0.7323,
        "Load Balance Std": 0.2046
    }
]

df = pd.DataFrame(results)

df.to_csv("../results/scheduler_comparison.csv", index=False)

print(df)
print("\nSaved to results/scheduler_comparison.csv")