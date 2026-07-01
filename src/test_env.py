from ppo_env import HPCSchedulingEnv

env = HPCSchedulingEnv()

state, _ = env.reset()

print("Initial state:", state)

state, reward, done, truncated, info = env.step(0)

print("Next state:", state)
print("Reward:", reward)
print("Done:", done)