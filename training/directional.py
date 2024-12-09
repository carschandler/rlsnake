# %%
import argparse
from pathlib import Path

import cli_directional
import numpy as np
import torch
from snake.envs import SnakeGrid
from snake.render.asciinema import render_trajectory
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictSequential as Seq
from torchrl.envs import (
    CatTensors,
    DTypeCastTransform,
    ExplorationType,
    FlattenObservation,
    GymEnv,
    StepCounter,
    TransformedEnv,
    UnsqueezeTransform,
    set_exploration_type,
)
from torchrl.modules import (
    MLP,
    Actor,
    ConvNet,
    DdpgCnnActor,
    DdpgCnnQNet,
    DuelingCnnDQNet,
    EGreedyModule,
    QValueActor,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# %%
args = cli_directional.parse_args()

# %%
env = GymEnv("snake/SnakeSurroundings", size=args.board_size, render_mode="ansi")
try:
    env.auto_register_info_dict()
except Exception:
    pass
env = TransformedEnv(env, CatTensors(["food", "danger"], "observation"))
env = TransformedEnv(
    env, DTypeCastTransform(torch.int8, torch.float32, in_keys="observation")
)
env = TransformedEnv(env, StepCounter(max_steps=args.max_episode_steps))

# %%
env.reset()

# %%
# value_net = ConvNet(strides=1, kernel_sizes=[3, 3, 2])
# policy = Seq(QValueActor(value_net, spec=env.action_spec))
# value_net = DdpgCnnQNet()
# policy = Actor(
#     DdpgCnnActor(
#         action_dim=env.action_spec.shape[-1], conv_net_kwargs={"kernel_sizes": 2}
#     )
# )
value_net = MLP(depth=2, num_cells=[64, 64], out_features=4)
policy = QValueActor(value_net, spec=env.action_spec)
policy(env.fake_tensordict())


env.reset()
rollout = env.rollout(max_steps=5, policy=policy)
env.reset()

# %%
exploration_module = EGreedyModule(
    env.action_spec,
    annealing_num_steps=args.buffer_length * args.optim_steps,
    eps_init=args.epsilon_bounds[0],
    eps_end=args.epsilon_bounds[1],
)

# policy_explore = Seq(policy, exploration_module)
policy_explore = torch.load("./output/surroundings_dqn_win/policy_explore.pt").cuda()
policy = policy_explore[0]

# %%
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer

collector = SyncDataCollector(
    env,
    policy_explore,
    frames_per_batch=args.steps_per_batch,
    total_frames=-1,
    init_random_frames=args.init_rand_steps,
    device=device,
)
rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_length, device=device))

from torch.optim import Adam

# %%
from torchrl.objectives import DQNLoss, SoftUpdate

loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True)
loss.make_value_estimator(gamma=args.gamma)
optim = Adam(loss.parameters(), lr=args.adam_learning_rate)
updater = SoftUpdate(loss, eps=0.99)

import time

# %%
from torchrl._utils import logger as torchrl_logger

path = Path("./output")

try:
    print(
        "Attempting to integrate with wandb; dismiss the following messages if you have"
        " not set it up."
    )
    from torchrl.record import WandbLogger

    logger = WandbLogger(
        project="rlsnake",
        exp_name=args.exp_name,
        offline=args.offline,
        tags=["surroundings", "dqn"] + args.tags,
        config=args,
    )
except Exception:
    print("wandb unavailable; falling back to CSV logging.")
    from torchrl.record import CSVLogger

    logger = CSVLogger(exp_name=args.exp_name, log_dir=str(path))
    logger.log_hparams(vars(args))

# %%
total_steps_in_training = 0
total_episodes = 0
t0 = time.time()
prev_max = 0
for i, data in enumerate(collector):
    print(f"{i}: len(rb)={len(rb)}, max={prev_max}, eps={exploration_module.eps}")
    # Write data in replay buffer
    rb.extend(data)

    total_steps_in_training += data.numel()
    total_episodes += data["next", "done"].sum()

    if len(rb) > args.init_rand_steps:
        # Optim loop (we do several optim steps
        # per batch collected for efficiency)
        for _ in range(args.optim_steps):
            sample = rb.sample(128)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
            # Update exploration factor
            # In theory, stepping by 100 each time and rb
            exploration_module.step(data.numel())
            # Update target params
            updater.step()
        if i % 10 == 0:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                n_rollout = 2000
                rollout: TensorDict = env.rollout(n_rollout, policy_explore, break_when_any_done=False)  # type: ignore
                max_snake_length = rollout["next", "snake_length"].max()
                max_step_count = rollout["next", "step_count"].max()
                print(
                    f"rollout({n_rollout}) max score: {max_snake_length}, max episode"
                    f" steps: {rollout['next', 'step_count'].max().item()}"
                )

                logger.log_scalar("Episode Score", max_snake_length)
                logger.log_scalar("Steps in Episode", max_step_count)
                logger.log_scalar("Total Training Steps", total_steps_in_training)
                logger.log_scalar("Total Episodes", total_episodes)
                logger.log_scalar("DQN Loss", loss_vals["loss"].item())
                logger.log_scalar("epsilon", exploration_module.eps)

                env.reset()

                if max_snake_length > prev_max and max_snake_length > 5:
                    i_max = rollout["next", "snake_length"].argmax()
                    i_start = rollout["next", "done"][:i_max].argwhere()
                    if i_start.numel() == 0:
                        i_start = 0
                    else:
                        i_start = i_start[-1][0] + 1
                    i_end = i_max + rollout["next", "done"][i_max:].argwhere()
                    if i_end.numel() == 0:
                        i_end = -1
                    else:
                        i_end = i_end[0, 0]

                    if rollout["next", "truncated"][i_end].item():
                        print(
                            "Warning: the trajectory which yielded a max score of"
                            f" {max_snake_length} was truncated at"
                            f" {args.max_episode_steps} steps."
                        )
                    logger.log_scalar("Overall Max Score", max_snake_length)

                    print(
                        f"New max of {max_snake_length}; rb ="
                        f" {len(rb):,}/{args.buffer_length:,}; eps ="
                        f" {exploration_module.eps}"
                    )
                    prev_max = max_snake_length

                    trajectory = rollout["next"][i_start : i_end + 1]
                    # trajectory = rollout["next"]

                    video_dir = path / args.exp_name / "videos"

                    video_dir.mkdir(parents=True, exist_ok=True)

                    render_trajectory(
                        str(video_dir / f"snake_length_{max_snake_length}.cast"),
                        SnakeGrid._render_as_string,
                        tensordict=trajectory.cpu(),
                    )

    if prev_max == args.board_size ** 2:
        break


t1 = time.time()

torchrl_logger.info(
    f"done after {total_steps_in_training} steps, {total_episodes} episodes and in"
    f" {t1-t0}s."
)

final_rollout = env.rollout(max_steps=10000, break_when_any_done=False, policy=policy)  # type: ignore

final_max_snake_length = final_rollout["next", "snake_length"].max()

if final_max_snake_length > prev_max:
    logger.log_scalar("Overall Max Score", final_max_snake_length)
    max_snake_length = final_max_snake_length

i_max = final_rollout["next", "snake_length"].argmax()
i_start = final_rollout["next", "done"][:i_max].argwhere()
if i_start.numel() == 0:
    i_start = 0
else:
    i_start = i_start[-1][0] + 1
i_end = i_max + final_rollout["next", "done"][i_max:].argwhere()
if i_end.numel() == 0:
    i_end = -1
else:
    i_end = i_end[0, 0]

trajectory = final_rollout["next"][i_start : i_end + 1]

video_dir = path / args.exp_name / "videos"

video_dir.mkdir(parents=True, exist_ok=True)

render_trajectory(
    str(video_dir / f"max_final_{final_max_snake_length}.cast"),
    SnakeGrid._render_as_string,
    tensordict=trajectory.cpu(),
)

torchrl_logger.info(
    f"Final rollout max score: {final_max_snake_length} in {len(trajectory)} steps"
)

torchrl_logger.info(f"Overall max score: {prev_max}")
