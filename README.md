# RL Snakes for ECE59500 RL Theory

## Setup

First, clone this repo.

### Weights and Biases (wandb)

Create a file called `.env` in the root of the repo. Enter the following lines
into it:

```
WANDB_ENTITY=<wandb-team-name>
WANDB_PROJECT=<wandb-project-name>
WANDB_BASE_URL=https://api.wandb.ai
WANDB_API_KEY=<wandb-api-key>
```

replacing the fields with the desired values.

## Running

### Docker

Build the docker image using

```
docker build -t rlsnakes .
```

Run the image interactively using

```
docker compose run rlsnakes
```

Once inside, use `python notebooks/training.py` to run RL training. Results will
populate into the `wandb` instance you set up above.

The directories `./lib ./notebooks ./scripts ./output` have been added as
bind-mounts in the docker-compose file, meaning you can edit them on your host
system and the changes will be reflected live in the docker image. Feel free to
update `docker-compose.yaml` with more if needed.
