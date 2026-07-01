from ppo_env_thermal import HPCSchedulingEnv

env = HPCSchedulingEnv()

state, _ = env.reset()

print("State length:", len(state))
print("Initial state:", state)

for _ in range(5):

    state, reward, done, truncated, info = env.step(0)

    print("Reward:", reward)

    for gpu in env.gpus:
        print(
            f"GPU{gpu.gpu_id} Temp = {gpu.temperature:.1f}"
        )