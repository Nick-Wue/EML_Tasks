import torch


## Tests the model

#  @param i_loss_func used loss function.

#  @param io_data_loader data loader containing the data to which the model is applied.

#  @param io_model model which is tested.

#  @return summed loss over all test samples, number of correctly predicted samples.

def test( i_loss_func,

          io_data_loader,

          io_model ):

  # switch model to evaluation mode

  io_model.eval()
  l_loss_total = 0
  l_n_correct = 0
  
  with torch.no_grad():
    for batch_ndx, sample in enumerate(io_data_loader):
      l_prediction = io_model(sample[0])
      l_loss_total += i_loss_func(l_prediction, sample[1]).item()
      l_n_correct += (l_prediction.argmax(1) == sample[1]).type(torch.float).sum().item()




  # TODO: finish implementation


  return l_loss_total, l_n_correct