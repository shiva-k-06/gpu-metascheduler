import math


class Simulator:

    def __init__(self, gpus, jobs):
        self.gpus = gpus
        self.jobs = jobs
        self.current_time = 0
        self.completed_jobs = []

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

    def update_running_jobs(self):
        for gpu in self.gpus:
            if not gpu.is_idle():
                gpu.remaining_time -= 1
                gpu.busy_time += 1

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

    def step(self):

       self.start_jobs_on_idle_gpus()
       self.update_running_jobs()

       self.current_time += 1