import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("../results/scheduler_comparison.csv")


def save_bar_chart(column, title, ylabel, filename):

    plt.figure(figsize=(8, 5))

    plt.bar(df["Scheduler"], df[column])

    plt.title(title)
    plt.xlabel("Scheduler")
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(f"../results/{filename}")
    plt.close()


save_bar_chart(
    "Makespan",
    "Makespan Comparison",
    "Time Units",
    "makespan_comparison.png"
)

save_bar_chart(
    "Avg Wait Time",
    "Average Wait Time Comparison",
    "Time Units",
    "wait_time_comparison.png"
)

save_bar_chart(
    "Avg GPU Utilization",
    "Average GPU Utilization Comparison",
    "Utilization Ratio",
    "utilization_comparison.png"
)

save_bar_chart(
    "Load Balance Std",
    "Load Balance Standard Deviation Comparison",
    "Std Dev",
    "load_balance_comparison.png"
)

print("Graphs saved in results folder.")