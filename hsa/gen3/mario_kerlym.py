from hsa.gen3.nes_env import NesEnv
from hsa.gen3.process import DynamicProxyProcess
from hsa.machine_constants import mario_rom_location, open_ai_gym_monitor_dir
from kerlym import agents
from kerlym.agents.dqn.networks import simple_dnn

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-e", "--env", dest="env", default="Mario-ram-v3",                  help="Which GYM Environment to run [%default]")
parser.add_option("-b", "--batch_size", dest="bs", default=120, type='int',              help="Batch size durring NN training [%default]")
parser.add_option("-o", "--dropout", dest="dropout", default=0.5, type='float',         help="Dropout rate in Q-Fn NN [%default]")
parser.add_option("-p", "--epsilon", dest="epsilon", default=0.2, type='float',         help="Exploration(1.0) vs Exploitation(0.0) action probability [%default]")
parser.add_option("-D", "--epsilon_decay", dest="epsilon_decay", default=1e-6, type='float',    help="Rate of epsilon decay: epsilon*=(1-decay) [%default]")
parser.add_option("-s", "--epsilon_min", dest="epsilon_min", default=0.05, type='float',help="Min epsilon value after decay [%default]")
parser.add_option("-d", "--discount", dest="discount", default=0.99, type='float',      help="Discount rate for future reards [%default]")
parser.add_option("-t", "--num_frames", dest="nframes", default=1, type='int',          help="Number of Sequential observations/timesteps to store in a single example [%default]")
parser.add_option("-m", "--max_mem", dest="maxmem", default=100000, type='int',         help="Max number of samples to remember [%default]")
parser.add_option("-P", "--plots", dest="plots", action="store_true", default=False,    help="Plot learning statistics while running [%default]")
parser.add_option("-F", "--plot_rate", dest="plot_rate", default=1, type='int',        help="Plot update rate in episodes [%default]")
parser.add_option("-a", "--agent", dest="agent", default="dqn",                         help="Which learning algorithm to use [%default]")
parser.add_option("-i", "--difference", dest="difference_obs", action="store_true", default=False,  help="Compute Difference Image for Training [%default]")
parser.add_option("-r", "--learning_rate", dest="learning_rate", type='float', default=1e-4,  help="Learning Rate [%default]")
parser.add_option("-R", "--render", dest="render", action='store_true', default=False,  help="Render game progress [%default]")
parser.add_option("-c", "--concurrency", dest="nthreads", type='int', default=1,  help="Number of Worker Threads [%default]")
(options, args) = parser.parse_args()


def nes_factory():
    return NesEnv(mario_rom_location, frameskip=4)


env_factory = lambda: DynamicProxyProcess(nes_factory)



agent = agents.DQN(
                    env=env_factory,
                    nthreads=1,
                    nframes=1,
                    epsilon=0.5,
                    discount=options.discount,
                    modelfactory=simple_dnn,
                    epsilon_schedule=lambda episode,epsilon: epsilon*(1-options.epsilon_decay),
                    batch_size=options.bs,
                    dropout=options.dropout,
                    stats_rate=options.plot_rate,
                    enable_plots = options.plots,
                    max_memory = options.maxmem,
                    difference_obs = options.difference_obs,
                    learning_rate=options.learning_rate,
                    preprocessor = None,
                    render = options.render
                    )
agent.train()
