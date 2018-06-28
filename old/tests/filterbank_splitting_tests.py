import unittest

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from ml_search import split_array

class TestSplittingMethods(unittest.TestCase):

    small = np.array([[ 1, 2, 3, 4],
                      [ 5, 6, 7, 8],
                      [ 9,10,11,12]])
    large = np.array([[ 1, 2, 3, 4, 5, 6, 7, 8],
                      [ 9,10,11,12,13,14,15,16],
                      [17,18,19,20,21,22,23,24],
                      [25,26,27,28,29,30,31,32]])

    def test_small(self):
        self.assertTrue(np.array_equal(split_array(self.small),
            [np.array([[ 1, 2, 3, 4],
                       [ 5, 6, 7, 8],
                       [ 9,10,11,12]])]
        ))

        self.assertTrue(np.array_equal(split_array(self.small,
                                              x_sample_num=2),
            [np.array([[ 1, 2],
                       [ 5, 6],
                       [ 9,10]]),
             np.array([[ 3, 4],
                       [ 7, 8],
                       [11,12]])]
        ))

        self.assertTrue(np.array_equal(split_array(self.small,
                                              y_sample_num=1),
            [np.array([[ 1, 2, 3, 4]]),
             np.array([[ 5, 6, 7, 8]]),
             np.array([[ 9,10,11,12]])]
        ))

        self.assertTrue(np.array_equal(split_array(self.small,
                                              x_sample_num=2,
                                              y_sample_num=1),
            [np.array([[ 1, 2]]),
             np.array([[ 3, 4]]),
             np.array([[ 5, 6]]),
             np.array([[ 7, 8]]),
             np.array([[ 9,10]]),
             np.array([[11,12]])]
        ))



if __name__ == '__main__':
    unittest.main()
