# Deep Reinforcement Learning with Varying Levels of Observability in _Snake_

![RL agent winning the snake game](./snake_win.gif)

Check out the final report at [`report/report.pdf`](report/report.pdf)

## Summary

This scope of this project was to study the effect of manipulating the
observation space (the space of possible observations that the agent makes on
the true state of the environment) on the performance and behavior of trained RL
policies. It includes a custom implementation of _Snake_ and four different
observation spaces using the [Gymnasium](https://gymnasium.farama.org/) API,
which are contained in [`lib/snake`](lib/snake). The policies are trained using
the new [TorchRL](https://pytorch.org/rl/stable/index.html) library: the
official RL component of the PyTorch ecosystem. I utilized the DDQN algorithm
paired with both MLPs and RNNs/CNNs to approximate value functions depending on
the observation space used. Results are optionally logged to a [Weights and
Biases](https://wandb.ai) instance for tracking experiment results and running
"sweeps" of parameter combinations.

## Setup

First, clone this repo.

### Environment Setup

You can run the code locally on your machine either inside a virtual environment
via Docker or on the host itself using Pixi. Running the Docker image is the
most portable option, but Pixi may work just as well for you.

You will probably have the smoothest experience on Linux, but Windows and OSX
should work as well.

#### Docker

Install the docker CLI if you haven't already (not docker desktop).

Build the docker image by running the following from the root of the repo

```
docker build -t rlsnake .
```

Create an environment variables file (populating it is optional)

```
touch .env
```

Run the image interactively using

```
docker compose run rlsnake
```

This should open you into a new shell on the docker instance at the `/workspace`
directory. Most of the directories in the repository have been added as
bind-mounts in the docker-compose file, meaning you can edit them on your host
system and the changes will be reflected live in the docker image and
vice-versa. Feel free to update `docker-compose.yaml` with more if needed.

If this does not work, try

```
docker run -it rlsnake
```

It will not add the directories as bind-mounts, but should still run the code.

#### Pixi

`pixi` is a package manager built on the conda ecosystem, but is much better
than conda for a number of reasons. It's from some of the team that created
`mamba`, which you may be familiar with.

Install `pixi` via the [official installation
instructions](https://pixi.sh/dev/) (copy and paste the one-line install
command).

Once you have access to the `pixi` command, you should be able to just run `pixi
install` inside the root directory of the repository to download all the
required dependencies.

From here, you can use `pixi run python` to access the environment's python
command, or you can drop into an interactive shell using `pixi shell` and just
use `python` inside the shell.

#### Weights and Biases (wandb)

*Note: this step is optional, and the code should switch over to logging output
via CSV files if it cannot connect to wandb.*

Results can be logged to a project on Weights and Biases, which will
automatically track training progress and generate plots.

First, you will need to create a project called "rlsnake" on your wandb account.

To connect to the wandb project, create a file called `.env` in the root of the
repo. Enter the following lines into it:

```
WANDB_ENTITY=<wandb-team-name>
WANDB_PROJECT=rlsnake
WANDB_BASE_URL=https://api.wandb.ai
WANDB_API_KEY=<wandb-api-key>
```

replacing the fields with the values corresponding to your wandb credentials.


## Running

Once your Python environment is set up, enter the `training` directory of the
cloned repo and then 

```
python directional.py <choose an experiment name here> --offline
```

to run RL training. You can replace `directional.py` with any of the scripts
that don't begin with `cli_`.

The `--offline` tells wandb not to try to connect you. If you don't provide
this, it will prompt you with options to either connect wandb on the spot or you
can choose "Don't visualize my results" to get the same effect as `--offline`.

Each of the training scripts has a corresponding CLI module dictating which
hyperparameters it accepts and defaults for them. The defaults have been set to
the parameters that yielded the highest scores during our testing, but modify
them as you wish. The only required argument is a name for your experiment,
which is arbitrary and up to your own choice.

To see the options available in the CLI, use:

```
python directional.py --help
```

## Results

Results will log to wandb if set up; otherwise they will log to `./output`.
Regardless of whether you are connected to wandb, animations of each new high
score beyond a score of 5 will log into the `./output/videos` directory. They
are written as `.cast` files which can be played live in the terminal using
`asciinema play <filename>.cast`.
