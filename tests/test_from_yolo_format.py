import unittest
from two_stage_model.dynamic_analysis import from_yolo_format

class TestFromYoloFormat(unittest.TestCase):
    def test_from_yolo_format(self):
        # Test with a simple bounding box in a 100x100 image
        xc, yc, bw, bh = 0.2, 0.3, 0.2, 0.2
        img_w, img_h = 100, 100
        x, y, xw, yh = from_yolo_format(xc, yc, bw, bh, img_w, img_h)
        self.assertEqual([x, y, xw, yh], [10, 20, 30, 40])

    def test_from_yolo_format_large_image(self):
        # Test with a bounding box in a larger image
        xc, yc, bw, bh = 0.125, 0.2, 0.15, 0.2
        img_w, img_h = 1000, 1000
        x, y, xw, yh = from_yolo_format(xc, yc, bw, bh, img_w, img_h)
        self.assertEqual([x, y, xw, yh], [50, 100, 200, 300])

    def test_from_yolo_format_boundary(self):
        # Test with a bounding box at the boundary of the image
        xc, yc, bw, bh = 0.5, 0.5, 1.0, 1.0
        img_w, img_h = 100, 100
        x, y, xw, yh = from_yolo_format(xc, yc, bw, bh, img_w, img_h)
        self.assertEqual([x, y, xw, yh], [0, 0, 100, 100])

    def test_from_yolo_format_zero_size(self):
        # Test with a zero-size bounding box
        xc, yc, bw, bh = 0.5, 0.5, 0.0, 0.0
        img_w, img_h = 100, 100
        x, y, xw, yh = from_yolo_format(xc, yc, bw, bh, img_w, img_h)
        self.assertEqual([x, y, xw, yh], [50, 50, 50, 50])

if __name__ == '__main__':
    unittest.main()
