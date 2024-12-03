# %%
import torch
from snake.envs import SnakeEnv
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictSequential as Seq
from torchrl.envs import (
    FlattenObservation,
    GymEnv,
    StepCounter,
    TransformedEnv,
    UnsqueezeTransform,
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

import render_asciinema

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# %%
size = 5
env = GymEnv("snake/Snake-v0", size=size)
env = TransformedEnv(env, UnsqueezeTransform(-3, in_keys=["observation"]))
env = TransformedEnv(env, StepCounter())
env.auto_register_info_dict()
# env = TransformedEnv(env, FlattenObservation(-2, -1, in_keys=["observation"]))

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
value_net = DuelingCnnDQNet(
    out_features=env.action_spec.shape[-1],
    out_features_value=1,
    cnn_kwargs={
        "kernel_sizes": [3, 3, 2],
        "strides": 1,
    },
    mlp_kwargs={
        "depth": 2,
        "num_cells": [
            64,
            64,
        ],
    },
    device=device,
)
policy = QValueActor(value_net, spec=env.action_spec)
policy(env.fake_tensordict())


rollout = env.rollout(max_steps=5, policy=policy)

# %%
buffer_length = 10_000_000

exploration_module = EGreedyModule(
    env.action_spec,
    annealing_num_steps=buffer_length,
    eps_init=0.1,
    eps_end=0.005,
)

policy_explore = Seq(policy, exploration_module)

# %%
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer

init_rand_steps = 5000
frames_per_batch = 100
optim_steps = 10
collector = SyncDataCollector(
    env,
    policy_explore,
    frames_per_batch=frames_per_batch,
    total_frames=-1,
    init_random_frames=init_rand_steps,
    device=device,
)
rb = ReplayBuffer(storage=LazyTensorStorage(buffer_length, device=device))

from torch.optim import Adam

# %%
from torchrl.objectives import DQNLoss, SoftUpdate

loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True)
loss.make_value_estimator(gamma=0.99)
optim = Adam(loss.parameters(), lr=0.002)
updater = SoftUpdate(loss, eps=0.99)

import time

# %%
from torchrl._utils import logger as torchrl_logger
from torchrl.record import CSVLogger

path = "./output/training_5x5"
logger = CSVLogger(exp_name="dqn", log_dir=path)

# %%
total_count = 0
total_episodes = 0
t0 = time.time()
prev_max = 0
for i, data in enumerate(collector):
    # Write data in replay buffer
    rb.extend(data)
    max_step_count = rb[:]["next", "step_count"].max()
    max_snake_length = rb[:]["next", "snake_length"].max()

    if max_snake_length > prev_max and max_snake_length > 5:
        prev_max = max_snake_length

        i_max = rb["next", "snake_length"].argmax()
        i_start = rb["next", "terminated"][:i_max].argwhere()[-1][0] + 1
        i_end = i_max + rb["next", "terminated"][i_max:].argwhere()[0, 0]

        trajectory = rb["next"][i_start : i_end + 1]

        render_asciinema.render_observations(
            trajectory["observation"],
            trajectory["current_direction"],
            trajectory["terminated"],
            f"{path}/dqn/videos/snake_length_{max_snake_length}.cast",
        )

    if len(rb) > init_rand_steps:
        # Optim loop (we do several optim steps
        # per batch collected for efficiency)
        for _ in range(optim_steps):
            sample = rb.sample(128)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
            # Update exploration factor
            exploration_module.step(data.numel())
            # Update target params
            updater.step()
            if i % 1000:
                # torchrl_logger.info(
                #     f"Max snake length: {max_snake_length}, Max steps:"
                #     f" {max_step_count}, rb length {len(rb)}"
                # )
                if i % 100000:
                    logger.log_scalar("max_score", max_snake_length)
                    logger.log_scalar("max_steps", max_step_count)
                    logger.log_scalar("total_count", total_count)
                    logger.log_scalar("total_episodes", total_episodes)

            total_count += data.numel()
            total_episodes += data["next", "done"].sum()
    if max_snake_length == 100:
        break

t1 = time.time()

torchrl_logger.info(
    f"solved after {total_count} steps, {total_episodes} episodes and in {t1-t0}s."
)
