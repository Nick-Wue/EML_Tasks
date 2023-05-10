import unittest
import math
from . import function
from . import context

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
