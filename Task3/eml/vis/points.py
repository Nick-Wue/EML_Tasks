import torch

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


## Plots the given points and colors them by the predicted labels.

#  It is assumed that a prediction larger than 0.5 corresponds to a red point.

#  All other points are black.

#  @param i_points points in R^3.

#  @param io_model model which is applied to derive the predictions.

def plot( i_points,

          io_model ):

  # switch to evaluation mode

  io_model.eval()


  with torch.no_grad():
    predictions = io_model(i_points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = [(0,0,0), [1,0,0]]
    cm = LinearSegmentedColormap.from_list("red_black", colors, N=2)

    ax.scatter(i_points.T[0], i_points.T[1], i_points.T[2], c=predictions, cmap=cm)
    plt.show()

    
