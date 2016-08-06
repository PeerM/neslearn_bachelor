import argparse
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    envarg = parser.add_argument_group('Environment')
    envarg.add_argument("--frame_skip", type=int, default=4, help="How many times to repeat each chosen action.")
    envarg.add_argument("--repeat_action_probability", type=float, default=0, help="Probability, that chosen action will be repeated. Otherwise random action is chosen during repeating.")
    # envarg.add_argument("--screen_width", type=int, default=0, help="Screen width after resize.")
    envarg.add_argument("--screen_height", type=int, default=2048, help="Screen height after resize.")

    memarg = parser.add_argument_group('Replay memory')
    memarg.add_argument("--replay_size", type=int, default=1000000, help="Maximum size of replay memory.")
    memarg.add_argument("--history_length", type=int, default=1, help="How many screen frames form a state.")

    netarg = parser.add_argument_group('Deep Q-learning network')
    netarg.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    netarg.add_argument("--discount_rate", type=float, default=0.9, help="Discount rate for future rewards.")
    netarg.add_argument("--batch_size", type=int, default=8, help="Batch size for neural network.")
    netarg.add_argument('--optimizer', choices=['rmsprop', 'adam', 'adadelta'], default='rmsprop', help='Network optimization algorithm.')
    netarg.add_argument("--decay_rate", type=float, default=0.95, help="Decay rate for RMSProp and Adadelta algorithms.")
    netarg.add_argument("--clip_error", type=float, default=1, help="Clip error term in update between this number and its negative.")
    netarg.add_argument("--target_steps", type=int, default=10000, help="Copy main network to target network after this many steps.")
    netarg.add_argument("--min_reward", type=float, default=-30, help="Minimum reward.")
    netarg.add_argument("--max_reward", type=float, default=10, help="Maximum reward.")
    netarg.add_argument("--batch_norm", type=str2bool, default=False, help="Use batch normalization in all layers.")

    #netarg.add_argument("--rescale_r", type=str2bool, help="Rescale rewards.")
    #missing: bufferSize=512,valid_size=500,min_reward=-1,max_reward=1

    neonarg = parser.add_argument_group('Neon')
    neonarg.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu', help='backend type')
    neonarg.add_argument('--device_id', type=int, default=0, help='gpu device id (only used with GPU backend)')
    neonarg.add_argument('--datatype', choices=['float16', 'float32', 'float64'], default='float32', help='default floating point precision for backend [f64 for cpu only]')
    neonarg.add_argument('--stochastic_round', const=True, type=int, nargs='?', default=False, help='use stochastic rounding [will round to BITS number of bits if specified]')

    antarg = parser.add_argument_group('Agent')
    antarg.add_argument("--exploration_rate_start", type=float, default=1, help="Exploration rate at the beginning of decay.")
    antarg.add_argument("--exploration_rate_end", type=float, default=0.1, help="Exploration rate at the end of decay.")
    antarg.add_argument("--exploration_decay_steps", type=float, default=1000000, help="How many steps to decay the exploration rate.")
    antarg.add_argument("--exploration_rate_test", type=float, default=0.05, help="Exploration rate used during testing.")
    antarg.add_argument("--train_frequency", type=int, default=4, help="Perform training after this many game steps.")
    antarg.add_argument("--train_repeat", type=int, default=1, help="Number of times to sample minibatch during training.")
    antarg.add_argument("--random_starts", type=int, default=30, help="Perform max this number of dummy actions after game restart, to produce more random game dynamics.")

    comarg = parser.add_argument_group('Common')
    comarg.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
    comarg.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Log level.")

    mainarg = parser.add_argument_group('Main loop')
    mainarg.add_argument("--save_weights_prefix",
                         help="Save network to given file. Epoch and extension will be appended.")
    return parser.parse_args(args)

