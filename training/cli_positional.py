import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("exp_name")
    parser.add_argument(
        "--offline",
        "-o",
        action="store_true",
        help=(
            "Stores the run locally with the option to sync it to wandb after the fact."
        ),
    )
    parser.add_argument(
        "--tags",
        "-t",
        nargs="*",
        default=[],
        help=(
            "Tags to add to the experiment in wandb. Can also be added in post through"
            " the web UI."
        ),
    )

    parser.add_argument(
        "--gamma",
        "-g",
        default=0.99,
        type=float,
        help="Discount factor to use in the return/value function calculations.",
    )

    parser.add_argument("--board-size", "-s", default=5, help="Playable board size.")
    parser.add_argument(
        "--max-episode-steps",
        "-m",
        default=2000,
        type=int,
        help="Maximum steps allowed in an episode before it is truncated.",
    )
    parser.add_argument(
        "--init-rand-steps",
        "-r",
        default=5000,
        type=int,
        help=(
            "How many random steps should be taken before engaging the exploration"
            " module."
        ),
    )
    parser.add_argument(
        "--adam-learning-rate",
        "-L",
        default=0.002,
        type=float,
        help="The learning rate to use in the ADAM optimizer.",
    )
    parser.add_argument(
        "--buffer-length",
        "-l",
        default=1_000_000,
        type=int,
        help="Length of the ReplayBuffer",
    )
    parser.add_argument(
        "--epsilon-bounds",
        "-e",
        nargs=2,
        type=float,
        default=[0.2, 0.0],
        help=(
            "Start and end values for the epsilon to use in the epsilon-greedy"
            " exploration module. Controls exploration vs. exploitation during"
            " training. Pass the high (starting) value first followed by the low (end)"
            " value."
        ),
    )
    parser.add_argument(
        "--steps-per-batch",
        "-b",
        default=100,
        type=int,
        help="How many steps to take in each batch from the data collector.",
    )

    parser.add_argument(
        "--total-steps",
        "-T",
        default=1_000_000,
        type=int,
        help="How many total steps to take.",
    )

    parser.add_argument(
        "--optim-steps",
        "-S",
        default=10,
        type=int,
        help="How many passes of the optimizer to make during each batch sample",
    )

    args = parser.parse_args()

    return args
