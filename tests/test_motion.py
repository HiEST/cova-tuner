import unittest

import numpy as np

import cova.motion.motion_detector as motion

class TestMotion(unittest.TestCase):

    def test_merge_all_boxes(self):
        input_boxes = np.array([
            [0, 0, 100, 100],
        ])
        self.assertListEqual(list(motion.merge_all_boxes(input_boxes)), list(input_boxes[0]))

        input_boxes = np.array([
            [0, 0, 100, 100],
            [90, 90, 110, 110]
        ])
        output_box = [0, 0, 110, 110]
        self.assertListEqual(list(motion.merge_all_boxes(input_boxes)), output_box)


if __name__ == '__main__':
    unittest.main()