import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader
import vis.fashion_mnist
from mlp.model import Model
import mlp.tester
import mlp.trainer


train_dataset = torchvision.datasets.FashionMNIST("~/eml/EML_Tasks/Task4/", download=True, transform=torchvision.transforms.ToTensor(), train=True)
dataloader_train = DataLoader(train_dataset, batch_size=64)


test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
print(len(test_data))
dataloader_test = DataLoader(test_data, batch_size=64)
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
print("Starting training")
for epoch in range(100):
    print("Training Loss:" + str(mlp.trainer.train(loss_func, dataloader_train, model, optimizer)))
    
    print(str("Total Loss: " +  str(mlp.tester.test(loss_func, dataloader_test, model)[0])) +
              "Correct Predictions: " + str(mlp.tester.test(loss_func, dataloader_test, model)[1]))
    if epoch % 10 == 0:
        vis.fashion_mnist.plot(0, 400, dataloader_test, model, epoch, i_path_to_pdf="out")