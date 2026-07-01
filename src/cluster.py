from gpu import GPU
from config import GPU_CONFIGS


def create_cluster():

    gpus = []

    for gpu_id, gpu_config in enumerate(GPU_CONFIGS):

        gpu = GPU(
            gpu_id=gpu_id,
            speed=gpu_config["speed"],
            memory=gpu_config["memory"]
        )

        gpus.append(gpu)

    return gpus