import hsa.gen3.nes_env
from extern.fceux_learningenv.nes_python_interface import NESInterface

nes_low = NESInterface("/home/peer/mario.nes", eb_compatible=False, auto_render_period=1)
nes_env = hsa.gen3.nes_env.NesEnv(frameskip=4, obs_type="image", nes=nes_low)
nes_env.reset()
for i in range(2000):
    nes_env.step(nes_env.action_space.sample())
