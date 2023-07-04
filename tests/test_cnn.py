'''
Unit testing - project for CMEPDA exam
Version 1.0
Owners: Giovanni & Ana
'''
import unittest
import numpy as np
from matplotlib import pyplot as plt

class Test(unittest.TestCase):

    '''Unit test for the CNN
    '''

    # Test shape
    def test_shape_train(self):
        self.assertEqual(len(l), 2 * mammo_o.shape[0])
        self.assertEqual(mammo_o.shape[0], mammo_f.shape[0])
        self.assertEqual(mammo_o.shape[0], label.shape[0])
        self.assertEqual(mammo_o.shape[0], mammo_o_4d.shape[0])
        self.assertEqual(mammo_o.shape[1], mammo_o_4d.shape[1])
        self.assertEqual(mammo_o.shape[2], mammo_o_4d.shape[2])
        self.assertEqual(mammo_f.shape[0], mammo_f_4d.shape[0])
        self.assertEqual(mammo_f.shape[1], mammo_f_4d.shape[1])
        self.assertEqual(mammo_f.shape[2], mammo_f_4d.shape[2])
        self.assertEqual(mammo_f.shape[0], mammo_f_4d.shape[0])
        self.assertEqual(mammo_o_4d.shape[3], 1)
        self.assertEqual(mammo_f_4d.shape[3], 1)
    
    def test_shape_test(self):
        self.assertEqual(len(l_t), 2 * mammo_o.shape[0])
        self.assertEqual(mammo_o_.shape[0], mammo_f_t.shape[0])
        self.assertEqual(mammo_o_t.shape[0], label_t.shape[0])
        self.assertEqual(mammo_o_t.shape[0], mammo_o_4d_t.shape[0])
        self.assertEqual(mammo_o_t.shape[1], mammo_o_4d_t.shape[1])
        self.assertEqual(mammo_o_t.shape[2], mammo_o_4d_t.shape[2])
        self.assertEqual(mammo_f_t.shape[0], mammo_f_4d_t.shape[0])
        self.assertEqual(mammo_f_t.shape[1], mammo_f_4d_t.shape[1])
        self.assertEqual(mammo_f_t.shape[2], mammo_f_4d_t.shape[2])
        self.assertEqual(mammo_f_t.shape[0], mammo_f_4d_t.shape[0])
        self.assertEqual(mammo_o_4d_t.shape[3], 1)
        self.assertEqual(mammo_f_4d_t.shape[3], 1)
    
    
if __name__ == '__main__':
    unittest.main()
