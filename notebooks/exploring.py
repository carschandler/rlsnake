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

# %%
