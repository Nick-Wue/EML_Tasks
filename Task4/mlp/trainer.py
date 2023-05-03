## Trains the given MLP-model.

#  @param i_loss_func used loss function.

#  @param io_data_loader data loader containing the data to which the model is applied (single epoch).

#  @param io_model model which is trained.

#  @param io_optimizer.

#  @return summed loss over all training samples.

def train( i_loss_func,

           io_data_loader,

           io_model,

           io_optimizer ):

  # switch model to training mode

  io_model.train()

  l_loss_total = 0

  
  for batch_ndx, sample in enumerate(io_data_loader):
    l_prediction = io_model(sample[0])
    l_loss_total = i_loss_func(l_prediction, sample[1])
    io_optimizer.zero_grad()
    l_loss_total.backward()
    io_optimizer.step()



  return l_loss_total.item()