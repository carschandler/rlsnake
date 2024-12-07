import multiprocessing
from pathlib import Path

import cli_rnn
import snake
import torch
import tqdm
from snake.envs import SnakeGridDiscrete
from snake.render.asciinema import render_trajectory
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictSequential
from tensordict.nn import TensorDictSequential as Seq
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import (
    Compose,
    DTypeCastTransform,
    ExplorationType,
    GrayScale,
    InitTracker,
    ObservationNorm,
    PermuteTransform,
    Resize,
    RewardScaling,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    UnsqueezeTransform,
    set_exploration_type,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ConvNet, EGreedyModule, LSTMModule, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate

args = cli_rnn.parse_args()

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

n_cells = 64  # HACK

lstm = LSTMModule(
    input_size=n_cells,
    hidden_size=32,
    device=device,
    in_key="embed",
    out_key="embed",
)

env = GymEnv("snake/SnakeGridDiscrete", size=args.board_size)
env.auto_register_info_dict()

transforms = Compose(
    StepCounter(max_steps=args.max_episode_steps),
    DTypeCastTransform(torch.int64, torch.float32, in_keys="observation"),
    InitTracker(),
)
env = TransformedEnv(env, transforms)
env.append_transform(lstm.make_tensordict_primer())
env = TransformedEnv(env, PermuteTransform(dims=[-1, -3, -2], in_keys="observation"))

feature = Mod(
    ConvNet(
        num_cells=[32, 32, 64],
        squeeze_output=True,
        aggregator_class=nn.AdaptiveAvgPool2d,
        aggregator_kwargs={"output_size": (1, 1)},
        device=device,
    ),
    in_keys=["observation"],
    out_keys=["embed"],
)


mlp = MLP(
    out_features=4,
    num_cells=[
        64,
    ],
    device=device,
)
mlp[-1].bias.data.fill_(0.0)
mlp = Mod(mlp, in_keys=["embed"], out_keys=["action_value"])

qval = QValueModule(spec=env.action_spec)

stoch_policy = Seq(feature, lstm, mlp, qval)

exploration_module = EGreedyModule(
    annealing_num_steps=args.annealing_steps,
    spec=env.action_spec,
    eps_init=args.epsilon_bounds[0],
    eps_end=args.epsilon_bounds[1],
)
stoch_policy = TensorDictSequential(
    stoch_policy,
    exploration_module,
)

policy = Seq(feature, lstm.set_recurrent_mode(True), mlp, qval)
policy(env.reset())

loss_fn = DQNLoss(policy, action_space=env.action_spec, delay_value=True)
loss_fn.make_value_estimator(gamma=args.gamma)

updater = SoftUpdate(loss_fn, eps=0.95)

optim = torch.optim.Adam(policy.parameters(), lr=args.adam_learning_rate)


collector = SyncDataCollector(
    env,
    stoch_policy,
    frames_per_batch=args.steps_per_batch,
    total_frames=args.total_steps,
)
rb = TensorDictReplayBuffer(
    storage=LazyMemmapStorage(args.buffer_length), batch_size=4, prefetch=10
)

pbar = tqdm.tqdm(total=collector.total_frames)
longest = 0

path = Path("./output")
try:
    print(
        "Attempting to integrate with wandb; dismiss the following messages if you have"
        " not set it up."
    )
    from torchrl.record import WandbLogger

    logger = WandbLogger(
        project="rlsnakes",
        exp_name=args.exp_name,
        offline=args.offline,
        tags=["rnn", "grid"] + args.tags,
        config=args,
    )
except Exception:
    print("wandb unavailable; falling back to CSV logging.")
    from torchrl.record import CSVLogger

    logger = CSVLogger(exp_name=args.exp_name, log_dir=str(path))
    logger.log_hparams(vars(args))

traj_lens = []
max_deterministic = 0
for i, data in enumerate(collector):
    pbar.update(data.numel())
    # it is important to pass data that is not flattened
    rb.extend(data.unsqueeze(0).to_tensordict().cpu())
    for _ in range(args.optim_steps):
        s = rb.sample().to(device, non_blocking=True)
        loss_vals = loss_fn(s)
        loss_vals["loss"].backward()
        optim.step()
        optim.zero_grad()
    longest = max(longest, data["next", "snake_length"].max().item())
    max_steps = data["next", "step_count"].max().item()
    pbar.set_description(
        f"max score: {longest}, loss_val: {loss_vals['loss'].item(): 4.4f}, eps:"
        f" {exploration_module.eps}"
    )
    exploration_module.step(data.numel())
    updater.step()

    if i % 10 == 0:
        logger.log_scalar(f"Max steps in batch of {args.steps_per_batch}", max_steps)
        logger.log_scalar("epsilon", exploration_module.eps)
        logger.log_scalar(f"Max Score Across All Training Steps", longest)
        logger.log_scalar("DQN Loss", loss_vals["loss"].item())
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            n_rollout = 1000
            rollout = env.rollout(n_rollout, stoch_policy)
            max_len = rollout.get(("next", "snake_length")).max().item()
            logger.log_scalar(
                f"Max deterministic score from rollout of {n_rollout} steps", max_len
            )
            if max_len > max_deterministic:
                max_deterministic = max_len

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
                        f" {max_len} was truncated at"
                        f" {args.max_episode_steps} steps."
                    )

                trajectory = rollout["next"][i_start : i_end + 1]
                # trajectory = rollout["next"]

                video_dir = path / args.exp_name / "videos"

                video_dir.mkdir(parents=True, exist_ok=True)

                render_trajectory(
                    str(video_dir / f"snake_length_{max_len}.cast"),
                    SnakeGridDiscrete._render_as_string_onehot,
                    tensordict=trajectory.cpu(),
                )

            traj_lens.append(max_len)
            env.reset()
