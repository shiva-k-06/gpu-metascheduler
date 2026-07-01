import math


class Simulator:

    def __init__(self, gpus, jobs):
        self.gpus = gpus
        self.jobs = jobs
        self.current_time = 0
        self.completed_jobs = []

        self.thermal_violations = 0
        self.thermal_violation_steps = 0
        self.max_temp_seen = 40.0
        self.temperature_sum = 0.0
        self.temperature_readings = 0

    def get_arriving_jobs(self):
        return [
            job for job in self.jobs
            if job.arrival_time == self.current_time
        ]

    def assign_job_to_gpu(self, job, gpu_id):
        self.gpus[gpu_id].queue.append(job)

    def start_jobs_on_idle_gpus(self):
        for gpu in self.gpus:
            if gpu.is_idle() and gpu.queue:
                job = gpu.queue.pop(0)

                gpu.running_job = job
                job.start_time = self.current_time
                gpu.remaining_time = math.ceil(job.runtime / gpu.speed)

                print(
                    f"Time {self.current_time}: "
                    f"Job{job.job_id} started on GPU{gpu.gpu_id}"
                )

    def update_temperature_metrics(self):
        for gpu in self.gpus:
            self.max_temp_seen = max(
                self.max_temp_seen,
                gpu.temperature
            )

            self.temperature_sum += gpu.temperature
            self.temperature_readings += 1

            if gpu.temperature > gpu.thermal_threshold:
                self.thermal_violation_steps += 1

                print(
                    f"[THERMAL WARNING] Time {self.current_time}: "
                    f"GPU{gpu.gpu_id} temperature = "
                    f"{gpu.temperature:.1f}°C "
                    f"(Threshold: {gpu.thermal_threshold}°C)"
                )

    def update_running_jobs(self):
        for gpu in self.gpus:

            if not gpu.is_idle():

                gpu.temperature = min(
                    95.0,
                    gpu.temperature + 0.15
                )

                if gpu.temperature > gpu.thermal_threshold:
                    if not gpu.was_overheated:
                        self.thermal_violations += 1
                        gpu.was_overheated = True
                else:
                    gpu.was_overheated = False

                gpu.busy_time += 1
                gpu.remaining_time -= 1

                if gpu.remaining_time <= 0:
                    job = gpu.running_job
                    job.finish_time = self.current_time

                    self.completed_jobs.append(job)

                    print(
                        f"Time {self.current_time}: "
                        f"Job{job.job_id} finished on GPU{gpu.gpu_id}"
                    )

                    gpu.running_job = None
                    gpu.remaining_time = 0

            else:
                gpu.temperature = max(
                    40.0,
                    gpu.temperature - 0.6
                )
                gpu.was_overheated = False

        self.update_temperature_metrics()

    def average_temperature(self):
        if self.temperature_readings == 0:
            return 0.0

        return self.temperature_sum / self.temperature_readings

    def step(self):
        self.start_jobs_on_idle_gpus()
        self.update_running_jobs()
        self.current_time += 1