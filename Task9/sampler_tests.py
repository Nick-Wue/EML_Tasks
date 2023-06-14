import torch


class SimpleDataSet( torch.utils.data.Dataset ):
  def __init__( self,
                i_length ):
    self.m_length = i_length

  def __len__( self ):
    return self.m_length

  def __getitem__( self,
                   i_idx ):
    return i_idx*10

test_dataset = SimpleDataSet(100)
sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,seed=42, rank=0, num_replicas=2, shuffle=True)
data_loader = torch.utils.data.DataLoader(test_dataset, sampler=sampler, batch_size=10)

# num_replicas splits data into n parts consisting of batch_size data samples
# rank determines which data samples are passed
# shuffle shuffles the data samples based on seed (as seen in the random order of printed indices)

print("Rank 0 of 2 replicas:")
for i in data_loader:
  print(i)

sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,seed=42, rank=1, num_replicas=2, shuffle=True)
data_loader = torch.utils.data.DataLoader(test_dataset, sampler=sampler, batch_size=10)

print("Rank 1 of 2 replicas:")
for i in data_loader:
  print(i)


# To demonstrate drop_last the data can't be split evenly
# the second batch is smaller so the data is distributed evenly
sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,seed=42, rank=0, num_replicas=2, shuffle=True, drop_last=True)
data_loader = torch.utils.data.DataLoader(test_dataset, sampler=sampler, batch_size=26)


print("Rank 0 of 2 replicas:")
for i in data_loader:
  print(len(i))

sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,seed=42, rank=1, num_replicas=2, shuffle=True)
data_loader = torch.utils.data.DataLoader(test_dataset, sampler=sampler, batch_size=26)

print("Rank 1 of 2 replicas:")
for i in data_loader:
  print(len(i))


batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=5, drop_last=False)
data_loader = torch.utils.data.DataLoader(test_dataset, sampler=batch_sampler, batch_size=10, drop_last=True)

# batches the data again into batches of size batch_size

for i in data_loader:
  print(i)