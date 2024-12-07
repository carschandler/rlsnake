import time


def render_trajectory(
    filepath: str,
    string_renderer,
    tensordict=None,
    observations=None,
    directions=None,
    done=None,
    won=None,
    snake_lengths=None,
    fps=5,
):
    with open(filepath, "w") as f:
        if tensordict is not None:
            observations = tensordict[
                "board" if "board" in tensordict else "observation"
            ]
            directions = tensordict["current_direction"]
            done = tensordict["done"]
            won = tensordict["won"]
            snake_lengths = tensordict["snake_length"]

        assert observations is not None
        assert directions is not None
        assert done is not None
        assert won is not None

        shape = observations.shape
        width = shape[-1] * 2 - 1
        height = shape[-2] + 1

        score_label = "Score: "
        score_string = lambda score: ("" if score is None else rf"Score: {score}\r\n")

        if snake_lengths is not None:
            height += 1
            width = max(
                width,
                len(score_label) + len(str(shape[-1] * shape[-2])),
            )

        f.write(
            rf'{{"version": 2, "width": {observations.shape[-1]}, "height":'
            rf' {observations.shape[-2]}, "timestamp": {int(time.time())}}}' + "\n"
        )
        t = 0
        for i, (observation, direction, done, won) in enumerate(
            zip(observations, directions, done, won)
        ):
            t += 1 / fps
            snake_length = None if snake_lengths is None else snake_lengths[i].item()

            board_string = string_renderer(
                observation.squeeze(), direction.item(), done.item(), won.item()
            )
            board_string = board_string.replace("\n", r"\r\n")
            f.write(
                rf'[{t}, "o",'
                rf' "\u001b[2J{board_string}\r\n{score_string(snake_length)}"]' + "\n"
            )
