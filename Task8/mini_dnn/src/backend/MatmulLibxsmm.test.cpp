#include <catch2/catch.hpp>
#include "MatmulLibxsmm.h"

TEST_CASE( "Tests the Matmul forward operator through LIBXSMM calls.",
           "[matmul][libxsmm][forward]" ) {
  // BLAS -> Deep Learning:
  // M: N (batch size)
  // K: C (in features)
  // N: K (out features)

  // sizes of the input
  int64_t l_size_n = 8;
  int64_t l_size_k = 16;
  int64_t l_size_c = 32;

  int64_t l_size_bn =  4;
  int64_t l_size_bk =  2;
  int64_t l_size_bc = 8;

  int64_t l_size_nb = l_size_n / l_size_bn;
  int64_t l_size_kb = l_size_k / l_size_bk;
  int64_t l_size_cb = l_size_c / l_size_bc;

  // construct input tensors
  at::Tensor l_x = at::rand( { l_size_n, l_size_c } );
  at::Tensor l_w = at::rand( { l_size_c, l_size_k } );

  // TODO:
  //   1) derive blocked X and W
  //   2) compute blocked solution through MatmulLibxsmm.forward
  //   3) reverse blocking and verify

  // X: nb x cb x bc x bn
  // W: kb x cb x bk x bc
  // Y: kb x nb x bk x bn

  at::Tensor l_x_v = l_x.view({l_size_nb, l_size_bn, l_size_cb, l_size_bc}).permute({0,2,3,1}).contiguous();
  at::Tensor l_w_v = l_w.view({l_size_cb, l_size_bc, l_size_kb, l_size_bk}).permute({2,0,3,1}).contiguous();

  mini_dnn::backend::MatmulLibxsmm matmul_o;
  at::Tensor l_y = matmul_o.forward(l_x_v, l_w_v);
  
  std::cout << "l_Y sizes:  " <<l_y.sizes() << std::endl;
  l_y = l_y.permute({1,3,0,2});
  l_y = l_y.contiguous();
  std::cout << "l_Y sizes:  " <<l_y.sizes() << std::endl;
  l_y = l_y.view({l_size_n, l_size_k});

  // compute reference
  std::cout << l_y;

  at::Tensor l_reference = at::matmul( l_x, l_w).contiguous();
  std::cout << l_reference;
  REQUIRE( at::allclose( l_y, l_reference ) );


}