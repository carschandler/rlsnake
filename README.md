# RL Snake for ECE59500 RL Theory

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

Build the docker image using

```
docker build -t rlsnake .
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

### Weights and Biases (wandb)

*Note: this step is optional, and the code should switch over to logging output
via CSV files if it cannot connect to wandb.*

Results can be logged to a project on Weights and Biases, which will
automatically track training progress and generate plots.

To connect to a wandb project, create a file called `.env` in the root of the
repo. Enter the following lines into it:

```
WANDB_ENTITY=<wandb-team-name>
WANDB_PROJECT=<wandb-project-name>
WANDB_BASE_URL=https://api.wandb.ai
WANDB_API_KEY=<wandb-api-key>
```

replacing the fields with the values corresponding to your wandb credentials.


## Running

Once your Python environment is set up, enter the `training` directory of the
cloned repo with and then use `python positional_dqn.py` to run RL training.

## Results

Results will log to wandb if set up; otherwise they will log to `./output`.
Regardless of whether you are connected to wandb, animations of each new high
score beyond a score of 5 will log into the `./output/videos` directory. They
are written as `.cast` files which can be played live in the terminal using
`asciinema play <filename>.cast`.
