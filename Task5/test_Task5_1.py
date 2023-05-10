import unittest
import Task5_1
class Test_Task5_1(unittest.TestCase):
    def test_forward_f(self):

        self.assertAlmostEqual(Task5_1.forward_f(4, 3, 4), 28.0)

    def test_backward_f(self):
        self.assertAlmostEqual(Task5_1.backward_f(3,5,2), [7, 3, 3])

    def test_forward_g(self):
        self.assertAlmostEqual(Task5_1.forward_g(5,3,3,4,9), 2.31e-16)
    
    def test_backward_g(self):
        l_dgdw0, l_dgdw1, l_dgdw2, l_dgdx0, l_dgdx1 = Task5_1.backward_g(5,3,3,4,9)
        self.assertAlmostEqual(l_dgdw0, 7.71e-22)
        self.assertAlmostEqual(l_dgdw1, 1.74e-21)
        self.assertAlmostEqual(l_dgdw2, 1.93e-22)
        self.assertAlmostEqual(l_dgdx0, 9.64e-22)
        self.assertAlmostEqual(l_dgdx1, 5.79e-22)

    def test_forward_h(self):
        self.assertAlmostEqual(Task5_1.forward_h(2,5), 4.215575456)

    def test_backward_h(self):
        l_dhdx, l_dhdy = Task5_1.backward_h(2,5)
        self.assertAlmostEqual(l_dhdx, -101.677514935)
        self.assertAlmostEqual(l_dhdy, -42.686757487)
        
