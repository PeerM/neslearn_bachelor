import os

from hsa.gen3.nes_env import NesEnv
from hsa.machine_constants import mario_rom_location, open_ai_gym_monitor_dir

env = NesEnv(mario_rom_location,frameskip=8)
env.monitor.start(os.path.join(open_ai_gym_monitor_dir,"./1"), force=True)
observation = env.reset()
reward_sum = 0
for _ in range(1000):
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  if done:
      env.reset()
  reward_sum += reward
print(reward_sum)
env.monitor.close()