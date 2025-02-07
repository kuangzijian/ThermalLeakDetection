import unittest
import numpy as np
from two_stage_model.dynamic_analysis import heatmap_eval

class TestHeatmapEval(unittest.TestCase):
    def test_heatmap_eval_above_threshold(self):
        heatmap = np.array([
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120],
            [130, 140, 150, 160]
        ])
        bbox = (1, 1, 3, 3)
        thresh = 60
        result = heatmap_eval(heatmap, bbox, thresh)
        self.assertTrue(result)

    def test_heatmap_eval_below_threshold(self):
        heatmap = np.array([
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120],
            [130, 140, 150, 160]
        ])
        bbox = (0, 0, 2, 2)
        thresh = 35
        result = heatmap_eval(heatmap, bbox, thresh)
        self.assertFalse(result)

    def test_heatmap_eval_equal_threshold(self):
        heatmap = np.array([
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120],
            [130, 140, 150, 160]
        ])
        bbox = (0, 0, 2, 2)
        thresh = 35
        result = heatmap_eval(heatmap, bbox, thresh)
        self.assertFalse(result)

    def test_heatmap_eval_empty_bbox(self):
        heatmap = np.array([
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120],
            [130, 140, 150, 160]
        ])
        bbox = (0, 0, 0, 0)
        thresh = 10
        result = heatmap_eval(heatmap, bbox, thresh)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
