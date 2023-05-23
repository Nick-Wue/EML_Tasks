#include <catch2/catch.hpp>
#include "MatmulAtenBlocked.h"

TEST_CASE( "Tests the Matmul forward operator through blocked Aten calls.",
           "[matmul][aten_blocked][forward]" ) {
  // BLAS -> Deep Learning:
  // M: N (batch size)
  // K: C (in features)
  // N: K (out features)

  // sizes of the input
  int64_t l_size_n = 128;
  int64_t l_size_k = 256;
  int64_t l_size_c = 512;

  int64_t l_size_bn =  64;
  int64_t l_size_bk =  32;
  int64_t l_size_bc = 128;

  int64_t l_size_nb = l_size_n / l_size_bn;
  int64_t l_size_kb = l_size_k / l_size_bk;
  int64_t l_size_cb = l_size_c / l_size_bc;

  // construct input tensors
  at::Tensor l_x = at::rand( { l_size_n, l_size_c } );
  at::Tensor l_w = at::rand( { l_size_c, l_size_k } );

  // TODO:
  //   1) derive blocked X and W
  //   2) compute blocked solution through MatmulAtenBlocked.forward
  //   3) reverse blocking and verify
  // X: nb x cb x bc x bn
  // W: kb x cb x bk x bc
  // Y: kb x nb x bk x bn

  //split in one dimension
  auto l_x_v = l_x.view({l_size_nb, l_size_bn, l_size_cb, l_size_bc}).permute({0,2,3,1}).contiguous();
  auto l_w_v = l_w.view({l_size_cb, l_size_bc, l_size_kb, l_size_bk}).permute({0,2,3,1}).contiguous();

  std::cout << l_x_v[0].sizes();
  std::cout << l_x_v[0].strides();
  std::cout << l_w_v[0].sizes();
  std::cout << l_w_v[0].strides();

  mini_dnn::backend::MatmulAtenBlocked matmul_o;
  auto l_y = matmul_o.forward(l_x_v, l_w_v);
  l_y = l_y.view({l_size_n, l_size_k});
  // compute reference

  at::Tensor l_reference = at::matmul( l_x, l_w );

  // check solution
  REQUIRE( at::allclose( l_y, l_reference ) );
}