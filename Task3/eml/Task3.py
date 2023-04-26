import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import perceptron.trainer
from perceptron.model import Model
import vis.points
from matplotlib.colors import LinearSegmentedColormap


#Task 3.1
data = np.genfromtxt("data_points.csv",delimiter=",")
data = torch.tensor(data, dtype=torch.float32)
labels = np.genfromtxt("data_labels.csv",delimiter=",")
labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = [(0,0,0), [1,0,0]]
cm = LinearSegmentedColormap.from_list("red_black", colors, N=2)

ax.scatter(data.T[0], data.T[1], data.T[2], c=labels, cmap=cm)

plt.show()

#Task 3.2
t_dataset = TensorDataset(torch.tensor(data), torch.tensor(labels))
# batch_size sets how many samples per batch
t_dataloader = DataLoader(t_dataset, batch_size=100)


# Task 3.3 + 3.4
model = Model()
loss_func = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1E-1)
for epoch in range(100):
    perceptron.trainer.train(loss_func, t_dataloader, model, optimizer)
    if epoch % 20 == 0:
        vis.points.plot(data, model)

