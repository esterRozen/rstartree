from unittest import TestCase
import random as rand
import numpy as np
from numpy.typing import NDArray

from src.rstartree import RStarTree, RSNode, BoundingBox
from src.boundingbox import Point


class TestRSNode(TestCase):
    @staticmethod
    def new():
        return RStarTree(1, 4)

    @staticmethod
    def new_big():
        return RStarTree(8, 40)

    def _is_leaf_check(self, node: RSNode):
        is_leaf = node.is_leaf
        has_node_child = False
        for child in node.children:
            if isinstance(child, RSNode):
                has_node_child = True
        if is_leaf == has_node_child:
            self.fail()
        return

    def _relation_check(self, node: RSNode):
        if not node.is_leaf:
            for child in node.children:
                if child.parent != node:
                    self.fail(f"parent {node} does not match child's "
                              f"({child}) parent {child.parent}")
        return

    def _bounds_verification(self, node: RSNode):
        if node.is_leaf:
            bounds = BoundingBox.create(node.children)
        else:
            list_bounds = list(map(lambda child: child.bounds, node.children))
            bounds = BoundingBox.create(list_bounds)
        self.assertTrue(bounds == node.bounds)

    def _traversal_tests(self, root: RSNode):
        self._bounds_verification(root)
        self._relation_check(root)
        self._is_leaf_check(root)

        if not root.is_leaf:
            for child in root.children:
                self._traversal_tests(child)

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

    def test_insert_point(self):
        """
        tests invariants of insertion.
        - bounds must remain consistent
        - relationships must be proper
        - must never be overfilled
        - must always have correct properties
        """
        tree = self.new()
        tree.insert(Point(np.array([1, 3])))
        self._traversal_tests(tree.root)
        tree.insert(Point(np.array([2, 2])))
        self._traversal_tests(tree.root)
        tree.insert(Point(np.array([5, 2])))
        self._traversal_tests(tree.root)
        tree.insert(Point(np.array([4, 3])))
        self._traversal_tests(tree.root)

        tree.insert(Point(np.array([2, 3])))
        self._traversal_tests(tree.root)
        tree.insert(Point(np.array([5, 4])))
        self._traversal_tests(tree.root)
        tree.insert(Point(np.array([8, 2])))
        self._traversal_tests(tree.root)
        tree.insert(Point(np.array([5, 6])))
        self._traversal_tests(tree.root)
        tree.insert(Point(np.array([4, 8])))
        self._traversal_tests(tree.root)

        tree.insert(Point(np.array([5, 5])))
        self._traversal_tests(tree.root)
        tree.insert(Point(np.array([6, 1])))
        self._traversal_tests(tree.root)
        tree.insert(Point(np.array([8, 5])))
        self._traversal_tests(tree.root)
        tree.insert(Point(np.array([7, 4])))
        self._traversal_tests(tree.root)

        tree.insert(Point(np.array([35, 31])))
        self._traversal_tests(tree.root)
        for _ in range(100):
            x = rand.randint(1, 50)
            y = rand.randint(1, 50)
            tree.insert(Point(np.array([x, y])))
            self._traversal_tests(tree.root)

    def test_insert_capacity_point(self):
        tree = self.new_big()

        for _ in range(10000):
            x = rand.randint(1, 50)
            y = rand.randint(1, 50)
            tree.insert(Point(np.array([x, y])))

        print(tree.height)
        self._traversal_tests(tree.root)
        tree.insert(Point(np.array([1, 1])))
        return

    def test_query_point(self):
        tree = self.new()

        for _ in range(100):
            x = rand.randint(1, 50)
            y = rand.randint(1, 50)
            tree.insert(Point(np.array([x, y])))

        tree.insert(Point(np.array([20, 22])))

        result = tree.query(Point(np.array([20, 22])))

        self.assertTrue(result is not None)
        self.assertTrue(result[0] == Point(np.array([20, 22])))

    def test_split(self):
        """
        tests the invariants of the split
        specifically ensure splits are chosen by correct heuristic
        """
        self.fail()


class TestRStarTree(TestCase):
    def test_insert(self):
        self.fail()
