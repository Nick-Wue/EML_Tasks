import torch

import matplotlib.pyplot as plt



## Converts an Fashion MNIST numeric id to a string.

#  @param i_id numeric value of the label.

#  @return string corresponding to the id.

def toLabel( i_id ):

  l_labels = [ "T-Shirt",

               "Trouser",

               "Pullover",

               "Dress",

               "Coat",

               "Sandal",

               "Shirt",

               "Sneaker",

               "Bag",

               "Ankle Boot" ]


  return l_labels[i_id]


## Applies the model to the data and plots the data.

#  @param i_off offset of the first image.

#  @param i_stride stride between the images.

#  @param io_data_loader data loader from which the data is retrieved.

#  @param io_model model which is used for the predictions.

#  @param i_path_to_pdf optional path to an output file, i.e., nothing is shown at runtime.

#  @param epoch number of epoch for file name

def plot( i_off,

          i_stride,

          io_data_loader,

          io_model,
          
          epoch,

          i_path_to_pdf = None
          
          ):

  # switch to evaluation mode

  io_model.eval()


  # create pdf if required

  if( i_path_to_pdf != None ):

    import matplotlib.backends.backend_pdf

    l_pdf_file = matplotlib.backends.backend_pdf.PdfPages( i_path_to_pdf )
  
  dataset = io_data_loader.dataset
  l_pdf_file = matplotlib.backends.backend_pdf.PdfPages( i_path_to_pdf + "_" + str(epoch) + ".pdf")
  fig = plt.figure()
  i = 1
  for indx, sample in enumerate(dataset):
    
    if (indx % i_stride == i_off):
      fig.add_subplot(5, 5, i)
      i += 1
      prediction = io_model(sample[0])
      prediction = prediction.argmax(1)
      plt.imshow(sample[0].squeeze())
      plt.title(toLabel(prediction))
  l_pdf_file.savefig(fig)


  # close pdf if required

  if( i_path_to_pdf != None ):

    l_pdf_file.close()