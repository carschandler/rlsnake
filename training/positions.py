import multiprocessing
import os
import tempfile
import uuid

import cli_positions
import snake
import torch
from snake.envs import SnakePositional
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.collectors import MultiaSyncDataCollector, SyncDataCollector
from torchrl.data import LazyMemmapStorage, MultiStep, TensorDictReplayBuffer
from torchrl.envs import (
    EnvCreator,
    ExplorationType,
    ParallelEnv,
    RewardScaling,
    StepCounter,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    CatFrames,
    Compose,
    GrayScale,
    ObservationNorm,
    Resize,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.modules import MLP, EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.record.loggers.csv import CSVLogger
from torchrl.trainers import (
    LogReward,
    Recorder,
    ReplayBufferTrainer,
    Trainer,
    UpdateWeights,
)

args = cli_positions.parse_args()

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

# the learning rate of the optimizer
lr = 2e-3
# weight decay
wd = 1e-5
# the beta parameters of Adam
betas = (0.9, 0.999)
# Optimization steps per batch collected (aka UPD or updates per data)
n_optim = 8
gamma = 0.99
tau = 0.02
total_frames = 5000
init_random_frames = 100
frames_per_batch = 32
batch_size = 32
buffer_size = min(total_frames, 100000)
num_workers = 2  # 8
num_collectors = 2  # 4
eps_greedy_val = 0.1
eps_greedy_val_env = 0.005
init_bias = 2.0


def make_env(
    parallel=False,
    obs_norm_sd=None,
    num_workers=1,
):
    if obs_norm_sd is None:
        obs_norm_sd = {"standard_normal": True}
    if parallel:

        def maker():
            return GymEnv(
                "snake/SnakePositions",
                render_mode="ansi",
                device=device,
            )

        base_env = ParallelEnv(
            num_workers,
            EnvCreator(maker),
            # Don't create a sub-process if we have only one worker
            serial_for_single=True,
        )
    else:
        base_env = GymEnv(
            "snake/SnakePositions",
            render_mode="ansi",
            device=device,
        )

    env = TransformedEnv(
        base_env,
        Compose(
            StepCounter(max_steps=2000),  # to count the steps of each trajectory
            ObservationNorm(in_keys=["observation"], **obs_norm_sd),
        ),
    )
    return env


def get_norm_stats():
    test_env = make_env()
    test_env.transform[-1].init_stats(
        num_iter=1000, cat_dim=0, reduce_dim=[-1, -2, -4], keep_dims=(-1, -2)
    )
    obs_norm_sd = test_env.transform[-1].state_dict()
    # let's check that normalizing constants have a size of ``[C, 1, 1]`` where
    # ``C=4`` (because of :class:`~torchrl.envs.CatFrames`).
    print("state dict of the observation norm:", obs_norm_sd)
    test_env.close()
    del test_env
    return obs_norm_sd


def make_model(dummy_env):
    net = MLP(num_cells=[64, 64], depth=2, activation_class=nn.ELU)
    actor = QValueActor(net, spec=dummy_env.action_spec).to(device)

    # init actor: because the model is composed of lazy conv/linear layers,
    # we must pass a fake batch of data through it to instantiate them.
    tensordict = dummy_env.fake_tensordict()
    actor(tensordict)

    # we join our actor with an EGreedyModule for data collection
    exploration_module = EGreedyModule(
        spec=dummy_env.action_spec,
        annealing_num_steps=args.total_frames,
        eps_init=args.epsilon_bounds[0],
        eps_end=args.epsilon_bounds[1],
    )
    actor_explore = TensorDictSequential(actor, exploration_module)

    return actor, actor_explore


def get_replay_buffer(buffer_size, n_optim, batch_size):
    replay_buffer = TensorDictReplayBuffer(
        batch_size=batch_size,
        storage=LazyMemmapStorage(buffer_size),
        prefetch=n_optim,
    )
    return replay_buffer


def get_collector(
    stats,
    num_collectors,
    actor_explore,
    frames_per_batch,
    total_frames,
    device,
):
    # We can't use nested child processes with mp_start_method="fork"
    if is_fork:
        cls = SyncDataCollector
        env_arg = make_env(parallel=True, obs_norm_sd=stats, num_workers=num_workers)
    else:
        cls = MultiaSyncDataCollector
        env_arg = [
            make_env(parallel=True, obs_norm_sd=stats, num_workers=num_workers)
        ] * num_collectors
    data_collector = cls(
        env_arg,
        policy=actor_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        # this is the default behavior: the collector runs in ``"random"`` (or explorative) mode
        exploration_type=ExplorationType.RANDOM,
        # We set the all the devices to be identical. Below is an example of
        # heterogeneous devices
        device=device,
        storing_device=device,
        split_trajs=False,
        postproc=MultiStep(gamma=gamma, n_steps=5),
    )
    return data_collector


def get_loss_module(actor, gamma):
    loss_module = DQNLoss(actor, delay_value=True)
    loss_module.make_value_estimator(gamma=gamma)
    target_updater = SoftUpdate(loss_module, eps=0.995)
    return loss_module, target_updater


stats = get_norm_stats()
test_env = make_env(parallel=False, obs_norm_sd=stats)
# Get model
actor, actor_explore = make_model(test_env)
loss_module, target_net_updater = get_loss_module(actor, gamma)

collector = get_collector(
    stats=stats,
    num_collectors=num_collectors,
    actor_explore=actor_explore,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    device=device,
)
optimizer = torch.optim.Adam(
    loss_module.parameters(), lr=lr, weight_decay=wd, betas=betas
)
exp_name = f"dqn_exp_{uuid.uuid1()}"
tmpdir = tempfile.TemporaryDirectory()
logger = CSVLogger(exp_name=exp_name, log_dir=tmpdir.name)

log_interval = 500

trainer = Trainer(
    collector=collector,
    total_frames=total_frames,
    frame_skip=1,
    loss_module=loss_module,
    optimizer=optimizer,
    logger=logger,
    optim_steps_per_batch=n_optim,
    log_interval=log_interval,
)

buffer_hook = ReplayBufferTrainer(
    get_replay_buffer(buffer_size, n_optim, batch_size=batch_size),
    flatten_tensordicts=True,
)
buffer_hook.register(trainer)
weight_updater = UpdateWeights(collector, update_weights_interval=1)
weight_updater.register(trainer)
trainer.register_op("post_steps", actor_explore[1].step, frames=frames_per_batch)
trainer.register_op("post_optim", target_net_updater.step)
log_reward = LogReward(log_pbar=True)
log_reward.register(trainer)

trainer.train()
