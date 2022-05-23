import numpy as np
from numpy.typing import NDArray
from unittest import TestCase

from src.rstartree import RStarTree, RSNode, BoundingBox
from src.boundingbox import Point


class TestRSNode(TestCase):
    @staticmethod
    def new():
        return RStarTree(1, 4)

    def test_is_leaf(self):
        self.fail()

    def _traverse_relation_check(self, root: RSNode):
        for child in root.children:
            if not root.is_leaf:
                if child.parent != root:
                    self.fail(f"parent {root} does not match child's "
                              f"({child}) parent {child.parent}")
                self._traverse_relation_check(child)

    def _bounds_verification(self, root: RSNode):
        if root.is_leaf:
            bounds = BoundingBox.create(root.children)
        else:
            list_bounds = list(map(lambda child: child.bounds, root.children))
            bounds = BoundingBox.create(list_bounds)
        self.assertTrue(bounds == root.bounds)

    def _traverse_bound_check(self, root: RSNode):
        self._bounds_verification(root)
        if not root.is_leaf:
            for child in root.children:
                self._traverse_bound_check(child)

    def _traverse_tests(self, root: RSNode):
        self._traverse_bound_check(root)
        self._traverse_relation_check(root)

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
        """
        tests invariants of insertion.
        # bounds must remain consistent
        # relationships must be proper
        # must never be overfilled
        """
        tree = self.new()
        tree.insert(Point(np.array([1, 3])))
        self._traverse_tests(tree.root)
        tree.insert(Point(np.array([2, 2])))
        self._traverse_tests(tree.root)
        tree.insert(Point(np.array([5, 2])))
        self._traverse_tests(tree.root)
        tree.insert(Point(np.array([4, 3])))
        self._traverse_tests(tree.root)

        tree.insert(Point(np.array([2, 3])))
        self._traverse_tests(tree.root)
        tree.insert(Point(np.array([5, 4])))
        self._traverse_tests(tree.root)
        tree.insert(Point(np.array([8, 2])))
        self._traverse_tests(tree.root)
        tree.insert(Point(np.array([5, 6])))
        self._traverse_tests(tree.root)
        tree.insert(Point(np.array([4, 8])))
        self._traverse_tests(tree.root)

        tree.insert(Point(np.array([5, 5])))
        self._traverse_tests(tree.root)
        tree.insert(Point(np.array([6, 1])))
        self._traverse_tests(tree.root)
        tree.insert(Point(np.array([8, 5])))
        self._traverse_tests(tree.root)
        tree.insert(Point(np.array([7, 4])))
        self._traverse_tests(tree.root)

    def test_split(self):
        """
        tests the invariants of the split
        # splits properly chosen by heuristic
        """
        self.fail()


class TestRStarTree(TestCase):
    def test_insert(self):
        self.fail()

    def test_overflow_treatment(self):
        self.fail()
