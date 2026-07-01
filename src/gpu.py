class GPU:

    def __init__(self, gpu_id, speed, memory):

        self.gpu_id = gpu_id
        self.speed = speed
        self.memory = memory

        self.temperature = 40.0
        self.thermal_threshold = 80.0
        self.utilization = 0.0

        self.running_job = None
        self.remaining_time = 0

        self.queue = []
        self.busy_time = 0
        
        self.was_overheated = False

    def is_idle(self):

        return self.running_job is None

    def __repr__(self):

        return (
            f"GPU{self.gpu_id} "
            f"(speed={self.speed}, "
            f"memory={self.memory}GB)"
        )
        
    