import unittest
from unittest.mock import patch
from two_stage_model.dynamic_analysis import calculate_mbr, PotentialLeaking

class TestCalculateMbr(unittest.TestCase):

    def test_calculate_mbr_basic(self):
        bbxs = [(1, 1, 2, 2), (2, 2, 3, 3)]
        contours = [((1, 1), (2, 2)), ((2, 2), (3, 3))]
        ext = 1
        x_bound = 10
        y_bound = 10


        mbrs = calculate_mbr(bbxs, contours, ext, x_bound, y_bound)

        expected_mbrs = [PotentialLeaking(0, 0, 4, 4, [((1, 1), (2, 2)), ((2, 2), (3, 3))])]
        self.assertEqual(len(mbrs), len(expected_mbrs))
        for mbr, expected in zip(mbrs, expected_mbrs):
            self.assertEqual(mbr.coords, expected.coords)
            self.assertEqual(mbr.ancestors, expected.ancestors)

    def test_calculate_mbr_with_trimming(self):
        bbxs = [(1, 1, 2, 2), (2, 2, 3, 3)]
        contours = [((1, 1), (2, 2)), ((2, 2), (3, 3))]
        ext = 1
        x_bound = 2
        y_bound = 2

        mbrs = calculate_mbr(bbxs, contours, ext, x_bound, y_bound)

        expected_mbrs = [PotentialLeaking(0, 0, 2, 2, [((1, 1), (2, 2)), ((2, 2), (3, 3))])]
        self.assertEqual(len(mbrs), len(expected_mbrs))
        for mbr, expected in zip(mbrs, expected_mbrs):
            self.assertEqual(mbr.coords, expected.coords)
            self.assertEqual(mbr.ancestors, expected.ancestors)

if __name__ == '__main__':
    unittest.main()
