import unittest
import math
import function
from context import Context

class TestNop( unittest.TestCase ):
  def test_backward( self ):

    l_grad = function.Nop.backward( None,
                                    5.0 )

    self.assertEqual( l_grad,
                      5.0 )

class TestAdd( unittest.TestCase ):
  def test_forward( self ):
    l_result = function.Add.forward( None,
                                     3.0,
                                     4.0 )
    
    self.assertAlmostEqual( l_result,
                            7.0 )
  
  def test_backward( self ):
    l_grad_a, l_grad_b = function.Add.backward( None,
                                                5.0 )
    
    self.assertEqual( l_grad_a, 5.0 )
    self.assertEqual( l_grad_b, 5.0 )

class TestMul(unittest.TestCase):
  
  def test_forward(self):
    i_ctx = Context()
    l_result = function.Mul.forward(i_ctx, 3, 8)
    self.assertAlmostEqual(l_result, 24)
    
  def test_backward(self):
    i_ctx = Context()
    _ = function.Mul.forward(i_ctx, 3, 8)
    l_grad_a, l_grad_b = function.Mul.backward(i_ctx, 5)
    self.assertAlmostEqual(l_grad_a, 40)
    self.assertAlmostEqual(l_grad_b, 15)