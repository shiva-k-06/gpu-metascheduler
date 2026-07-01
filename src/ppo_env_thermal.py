import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cluster import create_cluster
from workload import generate_jobs
from simulator_thermal import Simulator


class HPCSchedulingEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.num_gpus = 3

        self.action_space = spaces.Discrete(self.num_gpus)

        self.observation_space = spaces.Box(
            low=0,
            high=1000,
            shape=(17,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.gpus = create_cluster()
        self.jobs = generate_jobs()
        self.sim = Simulator(self.gpus, self.jobs)

        self.job_index = 0

        return self._get_state(), {}

    def _get_state(self):
        state = []

        for gpu in self.gpus:
            state.extend([
    len(gpu.queue),
    0 if gpu.is_idle() else 1,
    gpu.speed,
    gpu.memory,
    gpu.temperature
])

        if self.job_index < len(self.jobs):
            job = self.jobs[self.job_index]
            state.extend([
                job.runtime,
                job.memory_required
            ])
        else:
            state.extend([0, 0])

        return np.array(state, dtype=np.float32)
    
    def step(self, action):

        if self.job_index >= len(self.jobs):
            return self._get_state(), 0, True, False, {}

        job = self.jobs[self.job_index]

        self.sim.assign_job_to_gpu(job, action)
        self.sim.start_jobs_on_idle_gpus()
        self.sim.update_running_jobs()
        self.sim.current_time += 1

        gpu = self.gpus[action]

        queue_lengths = [
            len(g.queue) + (0 if g.is_idle() else 1)
            for g in self.gpus
        ]

        avg_queue = sum(queue_lengths) / len(queue_lengths)

        imbalance_penalty = sum(
            abs(q - avg_queue)
            for q in queue_lengths
        )

        reward = 0

        # Queue/load balancing
        reward -= len(gpu.queue) * 0.2
        reward -= imbalance_penalty * 1.0

        # Prefer faster GPUs slightly
        reward += gpu.speed * 0.05

        # Memory violation
        if job.memory_required > gpu.memory:
            reward -= 10

        # Strong thermal penalty
        if gpu.temperature > gpu.thermal_threshold:
            reward -= 20

        # Gradual temperature penalty even before threshold
        reward -= max(0, gpu.temperature - 50) * 1.0
        

        self.job_index += 1

        done = self.job_index >= len(self.jobs)

        return self._get_state(), reward, done, False, {}