import unittest
import sys
import os


from two_stage_model.dynamic_analysis import boxes_overlap


class TestBoxesOverlap(unittest.TestCase):

    def test_boxes_completely_overlap(self):
        box1 = (1, 2, 3, 4)
        box2 = (1, 2, 3, 4)
        self.assertTrue(boxes_overlap(box1, box2), "Boxes that completely overlap should return True")

    def test_boxes_do_not_overlap(self):
        box1 = (1, 2, 3, 4)
        box2 = (5, 6, 7, 8)
        self.assertFalse(boxes_overlap(box1, box2), "Boxes that do not overlap should return False")

    def test_boxes_touch_at_edge(self):
        box1 = (1, 1, 3, 3)
        box2 = (4, 4, 3, 3)
        self.assertFalse(boxes_overlap(box1, box2), "Boxes that touch at the edge should return False")

    def test_one_box_inside_another(self):
        box1 = (1, 1, 5, 5)
        box2 = (2, 2, 3, 3)
        self.assertTrue(boxes_overlap(box1, box2), "One box inside another should return True")

    def test_no_overlap_with_zero_area_box(self):
        box1 = (1, 1, 1, 1)
        box2 = (2, 2, 3, 3)
        self.assertFalse(boxes_overlap(box1, box2), "Boxes with zero area should not overlap")


if __name__ == '__main__':
    unittest.main()
