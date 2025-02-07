import unittest
from two_stage_model.dynamic_analysis import compute_iou

class TestComputeIoU(unittest.TestCase):
    def test_no_overlap(self):
        box1 = [0, 0, 1, 1]
        box2 = [2, 2, 3, 3]
        iou = compute_iou(box1, box2)
        self.assertEqual(iou, 0.0)

    def test_partial_overlap(self):
        box1 = [0, 0, 2, 2]
        box2 = [1, 1, 3, 3]
        iou = compute_iou(box1, box2)
        expected_iou = 1 / 7
        self.assertAlmostEqual(iou, expected_iou, places=6)

    def test_complete_overlap(self):
        box1 = [0, 0, 2, 2]
        box2 = [0, 0, 2, 2]
        iou = compute_iou(box1, box2)
        self.assertEqual(iou, 1.0)

    def test_box_within_another(self):
        box1 = [0, 0, 3, 3]
        box2 = [1, 1, 2, 2]
        iou = compute_iou(box1, box2)
        expected_iou = 1 / 9
        self.assertAlmostEqual(iou, expected_iou, places=6)

    def test_touching_edges(self):
        box1 = [0, 0, 2, 2]
        box2 = [2, 0, 4, 2]
        iou = compute_iou(box1, box2)
        self.assertEqual(iou, 0.0)

if __name__ == '__main__':
    unittest.main()