import random


from job import Job
from config import (
    NUM_JOBS,
    MIN_RUNTIME,
    MAX_RUNTIME,
    MIN_MEMORY,
    MAX_MEMORY,
    MAX_ARRIVAL_TIME
)


def generate_jobs():
    

    jobs = []

    for job_id in range(NUM_JOBS):

        runtime = random.randint(
            MIN_RUNTIME,
            MAX_RUNTIME
        )

        memory_required = random.randint(
            MIN_MEMORY,
            MAX_MEMORY
        )

        arrival_time = random.randint(
            0,
            MAX_ARRIVAL_TIME
        )

        compute_intensity = round(
            random.uniform(0.1, 1.0),
            2
        )

        jobs.append(
            Job(
                job_id=job_id,
                runtime=runtime,
                memory_required=memory_required,
                arrival_time=arrival_time,
                compute_intensity=compute_intensity
            )
        )

    jobs.sort(
        key=lambda job: job.arrival_time
    )

    return jobs