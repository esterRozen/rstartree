from unittest import TestCase
from src.boundingbox import BoundingBox
import numpy as np


class TestBoundingBox(TestCase):
    def test_create(self):
        self.fail()

    def test_margin_of(self):
        bb1 = BoundingBox(np.array([10, 10]), np.array([5, 5]))
        margin = BoundingBox.margin_of(bb1)
        self.assertTrue(margin == 20, f"2D Bbox margin was {margin} when it should be 20")

        bb1 = BoundingBox(np.array([10, 10, 10]), np.array([5, 5, 5]))
        margin = BoundingBox.margin_of(bb1)
        self.assertTrue(margin == 60, f"3D Bbox margin was {margin} when it should be 60")

        margin = BoundingBox.margin_of(None)
        self.assertTrue(margin == 0, f"margin of none should be 0")

    def test_overlap_sc(self):
        bb1 = BoundingBox(np.array([10, 10]), np.array([5, 5]))
        bb2 = BoundingBox(np.array([8, 8]), np.array([3, 3]))
        overlap = BoundingBox.overlap_of(bb1, bb2)
        cmp = BoundingBox(np.array([8, 8]), np.array([5, 5]))
        self.assertTrue(overlap == cmp, "bounding boxes should be same")

        bb2 = BoundingBox(np.array([5, 5]), np.array([0, 0]))
        overlap = BoundingBox.overlap_of(bb1, bb2)
        cmp = BoundingBox(np.array([5, 5]), np.array([5, 5]))
        self.assertTrue(overlap == cmp, "zero volume overlap with connected boxes")

        bb2 = BoundingBox(np.array([5, 4.5]), np.array([0, 0]))
        overlap = BoundingBox.overlap_of(bb1, bb2)
        self.assertTrue(overlap is None, "should return none when no overlap at all")

    def test_volume_of(self):
        self.fail()

    def test_asymmetry(self):
        self.fail()

    def test_center_along(self):
        self.fail()

    def test_width(self):
        self.fail()

    def test_center(self):
        self.fail()

    def test_volume(self):
        self.fail()

    def test_volume_diff(self):
        self.fail()

    def test_margin(self):
        self.fail()

    def test_margin_diff(self):
        self.fail()

    def test_overlap(self):
        self.fail()

    def test_overlap_margin(self):
        self.fail()

    def test_overlap_volume(self):
        self.fail()

    def test_min_bb_with(self):
        self.fail()

    def test_encloses(self):
        self.fail()
