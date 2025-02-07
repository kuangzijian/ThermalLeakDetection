import unittest
from two_stage_model.dynamic_analysis import to_yolo_format

class TestToYoloFormat(unittest.TestCase):
    def test_to_yolo_format(self):
        # Test with a simple bounding box in a 100x100 image
        x, y, xw, yh = 10, 20, 30, 40
        img_w, img_h = 100, 100
        xc, yc, bw, bh = to_yolo_format(x, y, xw, yh, img_w, img_h)
        self.assertAlmostEqual(xc, 0.2)
        self.assertAlmostEqual(yc, 0.3)
        self.assertAlmostEqual(bw, 0.2)
        self.assertAlmostEqual(bh, 0.2)

    def test_to_yolo_format_large_image(self):
        # Test with a bounding box in a larger image
        x, y, xw, yh = 50, 100, 200, 300
        img_w, img_h = 1000, 1000
        xc, yc, bw, bh = to_yolo_format(x, y, xw, yh, img_w, img_h)
        self.assertAlmostEqual(xc, 0.125)
        self.assertAlmostEqual(yc, 0.2)
        self.assertAlmostEqual(bw, 0.15)
        self.assertAlmostEqual(bh, 0.2)

    def test_to_yolo_format_boundary(self):
        # Test with a bounding box at the boundary of the image
        x, y, xw, yh = 0, 0, 100, 100
        img_w, img_h = 100, 100
        xc, yc, bw, bh = to_yolo_format(x, y, xw, yh, img_w, img_h)
        self.assertAlmostEqual(xc, 0.5)
        self.assertAlmostEqual(yc, 0.5)
        self.assertAlmostEqual(bw, 1.0)
        self.assertAlmostEqual(bh, 1.0)

    def test_to_yolo_format_zero_size(self):
        # Test with a zero-size bounding box
        x, y, xw, yh = 50, 50, 50, 50
        img_w, img_h = 100, 100
        xc, yc, bw, bh = to_yolo_format(x, y, xw, yh, img_w, img_h)
        self.assertAlmostEqual(xc, 0.5)
        self.assertAlmostEqual(yc, 0.5)
        self.assertAlmostEqual(bw, 0.0)
        self.assertAlmostEqual(bh, 0.0)

if __name__ == '__main__':
    unittest.main()
