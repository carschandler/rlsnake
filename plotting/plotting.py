import numpy as np
import pandas as pd
import plotly.graph_objects as go

simple = pd.read_csv("./wandb_simple_det_score.csv").set_index("Step")
grids = pd.read_csv("./wandb_grid_det_score.csv").set_index("Step")

positional = simple.iloc[:, 0].dropna()
for i in range(1, len(positional)):
    positional.iloc[i] = max(positional.iloc[i], positional.iloc[i - 1])

simple_x_ratio = 1089100 / positional.index.max()

directional = simple.iloc[:, 3].dropna()
for i in range(1, len(directional)):
    directional.iloc[i] = max(directional.iloc[i], directional.iloc[i - 1])

grid = grids.iloc[:, 0].dropna()
for i in range(1, len(grid)):
    grid.iloc[i] = max(grid.iloc[i], grid.iloc[i - 1])

grid_x_ratio = 200

discrete = grids.iloc[:, 3].dropna()
for i in range(1, len(discrete)):
    discrete.iloc[i] = max(discrete.iloc[i], discrete.iloc[i - 1])

plotvals = {
    "Directional": directional,
    "Positional": positional,
    "Grid": grid,
    "Discrete Grid": discrete,
}

fig = go.Figure(
    data=[
        go.Scatter(x=series.index * scale, y=series, name=name)
        for (name, series), scale in zip(
            plotvals.items(), [simple_x_ratio, simple_x_ratio, 200, 200]
        )
    ]
)

fig.update_layout(
    xaxis_title_text="Total Training Steps",
    yaxis_title_text="Max Score for Evaluated Policy",
)

fig.show(renderer="browser")
