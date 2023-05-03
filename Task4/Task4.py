import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader
import vis.fashion_mnist
from mlp.model import Model
import mlp.tester
import mlp.trainer


dataset = torchvision.datasets.FashionMNIST("~/eml/EML_Tasks/Task4/", download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


# with PdfPages('out.pdf') as pdf:
#     fig = plt.figure()
#     for img in range (9):
#         fig.add_subplot(3, 3, img + 1)
#         plt.imshow(dataset[img][0].squeeze())
#         plt.title(vis.fashion_mnist.toLabel(dataset[img][1]))
#     pdf.savefig(fig)
    
    
model = Model()
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1E-1)
print("Strating training")
for epoch in range(100):
    print(mlp.trainer.train(loss_func, dataloader, model, optimizer))
    print(mlp.tester.test(loss_func, dataloader, model))
    if epoch % 10 == 0:
        vis.fashion_mnist.plot(0, 600, dataloader, model, epoch, i_path_to_pdf="out")