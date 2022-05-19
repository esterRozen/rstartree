import numpy as np
from numpy.typing import NDArray
from unittest import TestCase

from src.rstartree import RStarTree
from src.boundingbox import Point


class TestRSNode(TestCase):
    @staticmethod
    def new():
        return RStarTree(1, 4)

    def test_is_leaf(self):
        self.fail()

    def test_choose_subtree(self):
        def test__check_coverage():
            self.fail()

        def test__find_growth_overlap_perimeter():
            self.fail()

        def test__check_node_one():
            self.fail()

        def test__consider_candidates():
            self.fail()

        def test__best_candidate():
            self.fail()

        test__check_coverage()
        test__find_growth_overlap_perimeter()
        test__check_node_one()
        test__consider_candidates()
        test__best_candidate()

        self.fail()

    def test_query(self):
        self.fail()

    def test_insert(self):
        tree = self.new()
        tree.insert(Point(np.array([1, 3])))
        tree.insert(Point(np.array([2, 2])))
        tree.insert(Point(np.array([5, 2])))
        tree.insert(Point(np.array([4, 3])))

        tree.insert(Point(np.array([2, 3])))
        tree.insert(Point(np.array([5, 4])))
        tree.insert(Point(np.array([8, 2])))
        tree.insert(Point(np.array([5, 6])))
        tree.insert(Point(np.array([4, 8])))
        self.fail()

    def test_choose_split(self):
        self.fail()

    def test_split(self):
        self.fail()

    def test__determine_dim(self):
        self.fail()

    def test__minimize_on(self):
        self.fail()

    def test__compute_wf(self):
        self.fail()


class TestRStarTree(TestCase):
    def test_insert(self):
        self.fail()

    def test_overflow_treatment(self):
        self.fail()
