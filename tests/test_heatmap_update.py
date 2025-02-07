import unittest
import numpy as np
from two_stage_model.dynamic_analysis import heatmap_update


class TestHeatmapUpdate(unittest.TestCase):
    def test_heatmap_update_gain(self):
        heatmap = np.zeros((10, 10), dtype=np.uint8)
        bbox = (2, 2, 5, 5)
        updated_heatmap = heatmap_update(heatmap, bbox)

        expected_heatmap = np.zeros((10, 10), dtype=np.uint8)
        expected_heatmap[2:5, 2:5] = 20

        np.testing.assert_array_equal(updated_heatmap, expected_heatmap)

    def test_heatmap_update_loss(self):
        heatmap = np.full((10, 10), 15, dtype=np.uint8)
        bbox = -1
        updated_heatmap = heatmap_update(heatmap, bbox)

        expected_heatmap = np.full((10, 10), 5, dtype=np.uint8)

        np.testing.assert_array_equal(updated_heatmap, expected_heatmap)

    def test_heatmap_update_below_zero(self):
        heatmap = np.full((10, 10), 5, dtype=np.uint8)
        bbox = -1
        updated_heatmap = heatmap_update(heatmap, bbox)

        expected_heatmap = np.zeros((10, 10), dtype=np.uint8)

        np.testing.assert_array_equal(updated_heatmap, expected_heatmap)

    def test_heatmap_update_above_max(self):
        heatmap = np.zeros((10, 10), dtype=np.uint8)
        bbox = (0, 0, 10, 10)
        updated_heatmap = heatmap_update(heatmap, bbox)
        updated_heatmap = heatmap_update(updated_heatmap, bbox)
        updated_heatmap = heatmap_update(updated_heatmap, bbox)
        updated_heatmap = heatmap_update(updated_heatmap, bbox)
        updated_heatmap = heatmap_update(updated_heatmap, bbox)
        updated_heatmap = heatmap_update(updated_heatmap, bbox)
        updated_heatmap = heatmap_update(updated_heatmap, bbox)
        updated_heatmap = heatmap_update(updated_heatmap, bbox)
        updated_heatmap = heatmap_update(updated_heatmap, bbox)
        updated_heatmap = heatmap_update(updated_heatmap, bbox)
        updated_heatmap = heatmap_update(updated_heatmap, bbox)
        updated_heatmap = heatmap_update(updated_heatmap, bbox)

        expected_heatmap = np.full((10, 10), 255, dtype=np.uint8)

        np.testing.assert_array_equal(updated_heatmap, expected_heatmap)


if __name__ == '__main__':
    unittest.main()
