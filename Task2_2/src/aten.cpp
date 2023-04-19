#include <cstdlib>
#include <ATen/ATen.h>
#include <iostream>

int main() {
  std::cout << "running the ATen examples" << std::endl;

  float l_data[4*2*3] = {  0.0f,  1.0f,  2.0f, 
                           3.0f,  4.0f,  5.0f,

                           6.0f,  7.0f,  8.0f, 
                           9.0f, 10.0f, 11.0f,
                           
                          12.0f, 13.0f, 14.0f,
                          15.0f, 16.0f, 17.0f,
                          
                          18.0f, 19.0f, 20.0f,
                          21.0f, 22.0f, 23.0f };

  std::cout << "l_data (ptr): " << l_data << std::endl;

  // TODO: Add ATen code

  // Task 1.2
  at::Tensor l_tensor = at::from_blob(l_data, {4,2,3});

  // Task 1.3
  std::cout << "tensor: " << l_tensor << std::endl; 
  std::cout << "dtype: " << l_tensor.dtype() << std::endl;
  std::cout << "sizes: " << l_tensor.sizes() << std::endl;
  std::cout << "strides: " << l_tensor.strides() << std::endl;
  std::cout << "storage_offset: " << l_tensor.storage_offset() << std::endl; 
  std::cout << "devcice: " << l_tensor.device() << std::endl;
  std::cout << "layout: " << l_tensor.layout() << std::endl;
  std::cout << "data_ptr: " << l_tensor.data_ptr() << std::endl;
  std::cout << "is contiguous: " << l_tensor.is_contiguous() << std::endl;

  // Task 1.4
  // change 1.0f -> 2.0f
  l_data[1] = 2.0f;
  std::cout << "tensor: " << l_tensor << std::endl; 

  // change 2.0f -> 3.0f at the same position
  l_tensor[0][0][1] = 3.0f;
  std::cout << "tensor: " << l_tensor << std::endl; 

  // Task 1.5
  // Changes to the view apply changes to the tensor so they use the same memory for their data
  auto l_view = l_tensor.select(1,1);
  std::cout << l_view;
  l_view[0][1] = 100.0f;
  std::cout << l_view;
  std::cout << l_tensor;

  // Task 1.6
  //stride and sizes are changed so that the tensor can be contiguous in memory
  at::Tensor l_cont = l_view.contiguous();
  std::cout << "Contiguous: " <<l_cont;

  std::cout << "sizes: " << l_cont.sizes() << std::endl;
  std::cout << "strides: " << l_cont.strides() << std::endl;
  std::cout << "View: " <<l_view;
    std::cout << "sizes: " << l_view.sizes() << std::endl;
  std::cout << "strides: " << l_view.strides() << std::endl;
    // Task 2.1
  at::Tensor A = at::rand({16,4});
  at::Tensor B = at::rand({4,16});
  std::cout << A;
  std::cout << B;

  //Task 2.2
  std::cout << at::matmul(A,B);
  
  // Task 2.3
  at::Tensor T_0 = at::rand({16,4,2});  
  at::Tensor T_1 = at::rand({16,2,4});
  
  // Task 2.4
  std::cout << at::bmm(T_0, T_1);



  std::cout << "finished running ATen examples" << std::endl;




  return EXIT_SUCCESS;
}

