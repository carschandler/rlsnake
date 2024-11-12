# %%
from torchrl.envs import GymEnv

env = GymEnv("Pendulum-v1")

# %%
reset = env.reset()
reset

# %%
reset_with_action = env.rand_action(reset)
reset_with_action

# %%
stepped_data = env.step(reset_with_action)
stepped_data

# %%
from torchrl.envs import step_mdp

data = step_mdp(stepped_data)
data

# %%
rollout = env.rollout(max_steps=10)
rollout

# %%
transition = rollout[3]
transition

### https://pytorch.org/rl/stable/tutorials/getting-started-1.html

# %%
import torch
from tensordict.nn import TensorDictModule

env = GymEnv("Pendulum-v1")
module = torch.nn.LazyLinear(out_features=env.action_spec.shape[-1])
policy = TensorDictModule(
    module,
    in_keys=["observation"],
    out_keys=["action"],
)

# %%
rollout = env.rollout(max_steps=10, policy=policy)
rollout

# %%
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Normal
from torchrl.modules import ProbabilisticActor
from torchrl.modules import MLP

backbone = MLP(in_features=3, out_features=2)
extractor = NormalParamExtractor()
module = torch.nn.Sequential(backbone, extractor)
td_module = TensorDictModule(module, in_keys=["observation"], out_keys=["loc", "scale"])
policy = ProbabilisticActor(
    td_module,
    in_keys=["loc", "scale"],
    out_keys=["action"],
    distribution_class=Normal,
    return_log_prob=True,
)

rollout = env.rollout(max_steps=10, policy=policy)
rollout

# %%
