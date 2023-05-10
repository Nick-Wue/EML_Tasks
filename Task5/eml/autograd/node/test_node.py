import unittest
import math
import node
 
class TestNode(unittest.TestCase):
    def test_Node(self):
        i_node_x = node.Node(3)
        i_node_y = node.Node(5)
        i_node_z = node.Node(2)
        i_node_a = i_node_y + i_node_z
        i_node_b = i_node_a * i_node_x
        
        i_node_b.backward(1)
        
        self.assertAlmostEqual(i_node_b.m_value, 21)
        self.assertAlmostEqual(i_node_x.m_grad, 7)
        self.assertAlmostEqual(i_node_y.m_grad, 3)
        self.assertAlmostEqual(i_node_z.m_grad, 3)