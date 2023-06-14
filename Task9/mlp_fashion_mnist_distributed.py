import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader
import vis.fashion_mnist
from mlp.model import Model
import mlp.tester
import time
import mlp.trainer

torch.distributed.init_process_group("mpi")
rank = torch.distributed.get_rank()
i_size_distributed = torch.distributed.get_world_size()

train_dataset = torchvision.datasets.FashionMNIST("~/eml/EML_Tasks/Task4/", download=True, transform=torchvision.transforms.ToTensor(), train=True)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=i_size_distributed, rank=rank, shuffle=True, seed=42, drop_last=True)
dataloader_train = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)


test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
print(len(test_data))
test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, num_replicas=i_size_distributed, rank=rank, shuffle=True, seed=42, drop_last=True)
dataloader_test = DataLoader(test_data, batch_size=64, sampler= test_sampler)
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
training_loss = 0
total_loss = 0
correct_predictions = 0
print("Starting training")
t_0 = time.time()
for epoch in range(100):
    training_loss = mlp.trainer.train(loss_func, dataloader_train, model, optimizer)
    for l_pa in model.parameters():
        torch.distributed.all_reduce( l_pa.grad.data,
        op = torch.distributed.ReduceOp.SUM )
        l_pa.grad.data = l_pa.grad.data / float(i_size_distributed)
    torch.distributed.all_reduce(training_loss, op = torch.distributed.ReduceOp.SUM)
    training_loss = training_loss / float(i_size_distributed)
    #only print on one Rank to avoid multiple outputs
    if rank == 0:
        print("Training Loss:" +  str(training_loss))
    total_loss = mlp.tester.test(loss_func, dataloader_test, model)[0]
    correct_predictions = mlp.tester.test(loss_func, dataloader_test, model)[1]
    torch.distributed.all_reduce( total_loss,
        op = torch.distributed.ReduceOp.SUM )
    torch.distributed.all_reduce( correct_predictions,
        op = torch.distributed.ReduceOp.SUM )
    total_loss = total_loss / float(i_size_distributed)
    
    if rank == 0:
        print(str("Total Loss: " +  str(total_loss)) +
              "Correct Predictions: " + str(correct_predictions))
    
t_1 = time.time()
torch.distributed.all_reduce( t_0,
        op = torch.distributed.ReduceOp.MIN )
torch.distributed.all_reduce( t_1,
        op = torch.distributed.ReduceOp.MAX )
if rank == 0:
    print(t_1 -t_0)