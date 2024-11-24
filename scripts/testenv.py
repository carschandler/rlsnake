import gymnasium, snake
from snake.envs.snake_env import Actions, Directions
from snake.envs import SnakeEnv

env = gymnasium.make("snake/Snake-v0", render_mode="ansi")

obs, info = env.reset()

print(info)


def r():
    print(env.render())


r()

se = SnakeEnv()

# text_to_action = dict(
#     cw=Actions.TURN_CW,
#     ccw=Actions.TURN_CCW,
#     s=Actions.STRAIGHT,
# )

text_to_action = dict(
    w=Actions.UP,
    a=Actions.LEFT,
    s=Actions.DOWN,
    d=Actions.RIGHT,
    h=Actions.LEFT,
    j=Actions.DOWN,
    k=Actions.UP,
    l=Actions.RIGHT,
)

term = False
while not term:
    s = input(f"Enter action ({', '.join(text_to_action.keys())}, exit): ")

    if s == "exit":
        break

    if s not in text_to_action:
        continue

    _, _, term, _, info = env.step(text_to_action[s].value)
    print(info)
    r()
