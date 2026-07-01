from cluster import create_cluster
from workload import generate_jobs
from simulator import Simulator


gpus = create_cluster()
jobs = generate_jobs()

sim = Simulator(gpus, jobs)

while len(sim.completed_jobs) < len(jobs):
    sim.step()

print("\nSimulation finished")
print("Total time:", sim.current_time)
print("Completed jobs:", len(sim.completed_jobs))