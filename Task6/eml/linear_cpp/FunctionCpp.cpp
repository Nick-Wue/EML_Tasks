#include <pybind11/pybind11.h>
#include <torch/extension.h>


torch::Tensor mm_multiplication(torch::Tensor left, torch::Tensor right){
  int rows_l = left.sizes()[1];
  int cols_r = right.sizes()[0];
  int rows_r = right.sizes()[1];
  torch::Tensor res = torch::zeros({rows_l, cols_r});

  for (int i = 0; i < rows_l; i++) {
    for (int j = 0; j < cols_r; j++) {
        for (int k = 0; k < rows_r; k++) {
            res[i][j] += left[i][k] * right[k][j];
          }
    }
  }
  return res;
}

torch::Tensor forward( torch::Tensor i_input, torch::Tensor i_weight){
  auto res = mm_multiplication(i_input, i_weight.transpose(0,1));
  return res;

}

std::vector< torch::Tensor > backward( torch::Tensor i_grad,
                                       torch::Tensor i_input,
                                       torch::Tensor i_weights ){
                                        std::vector<torch::Tensor> result; 
                                        auto grad_input = mm_multiplication(i_grad, i_weights);
                                        auto grad_weights = mm_multiplication(i_grad.transpose(0,1), i_input);
                                        result.push_back(grad_input);
                                        result.push_back(grad_weights);
                                        return result;

                                       }

PYBIND11_MODULE( TORCH_EXTENSION_NAME,
                 io_module) {
  io_module.def( "forward",
                 &forward,
                 "Forward pass of the Layer");
  io_module.def("backward",
                &backward,
                "Backward pas of the Layer");
  io_module.def("mm_multiplication",
                &mm_multiplication,
                "Simple matrix matrix multiplication");
}