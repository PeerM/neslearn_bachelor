from hsa.gen3.fceux_process import fceux_process_factory
from hsa.gen3.nes_env import NesEnv
from hsa.machine_constants import mario_rom_location, open_ai_gym_monitor_dir

isolate_in_process = True

if isolate_in_process:
    env = fceux_process_factory(mario_rom_location, frameskip=8)
else:
    env = NesEnv(frameskip=8, game_path=mario_rom_location)
# env.monitor.start(os.path.join(open_ai_gym_monitor_dir,"./1"), force=True)
observation = env.reset()
env.step(0)
reward_sum = 0
for _ in range(120):
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    env.render()
    if done:
        env.reset()
    reward_sum += reward
print(reward_sum)
env.close()
del env