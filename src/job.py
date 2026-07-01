class Job:

    def __init__(
        self,
        job_id,
        runtime,
        memory_required,
        arrival_time,
        compute_intensity
    ):

        self.job_id = job_id

        self.runtime = runtime
        self.memory_required = memory_required
        self.arrival_time = arrival_time
        self.compute_intensity = compute_intensity

        # Filled during simulation
        self.start_time = None
        self.finish_time = None

    def __repr__(self):

        return (
            f"Job{self.job_id} "
            f"(runtime={self.runtime}, "
            f"memory={self.memory_required}GB, "
            f"intensity={self.compute_intensity})"
        )