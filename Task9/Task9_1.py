import torch.distributed

torch.distributed.init_process_group("mpi")

rank = torch.distributed.get_rank()
size = torch.distributed.get_world_size()

print("Rank and size: ")
print(rank, size)

tensor = torch.empty((3,4))


# Blocking send and receive
if rank == 0:
    tensor= torch.ones_like(tensor)
    torch.distributed.send(tensor, 1)

else:
    if rank == 1:
        torch.distributed.recv(tensor, 0)
    tensor = torch.zeros_like(tensor)


# Non-blocking send and receive
if rank == 0:
    req = torch.distributed.isend(tensor, 1)

if rank == 1:
    req = torch.distributed.irecv(tensor, 0)

req.wait()

tensor = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

torch.distributed.all_reduce(tensor, torch.distributed.ReduceOp.SUM)

# only print once
if rank == 0:
    print(tensor)
