#! /usr/bin/env python3
"""
Rights to usage of this implementation and derivative works mine.
Non-commercial rights granted, but may not deviate from same
license in works based on this.

Algorithm based on R* Tree improved method:
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.367.7273&rep=rep1&type=pdf
as well as original algorithm:
https://infolab.usc.edu/csci599/Fall2001/paper/rstar-tree.pdf
some r* tree algos which weren't majorly different
from their r tree counterpart loosely based on:
http://www.mathcs.emory.edu/~cheung/Courses/554/Syllabus/3-index/R-tree3.html
"""
import math
from typing import List, Union, Tuple
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

eps = 0.001


@dataclass
class BoundingBox:
    # points are abstracted as same top and bottom bounds
    tops: NDArray[float]
    bottoms: NDArray[float]

    def __repr__(self):
        return "tops: " + self.tops.__repr__()[6:-1] + ", bottoms: " + self.bottoms.__repr__()[6:-1]

    def __copy__(self) -> 'BoundingBox':
        return BoundingBox(self.tops, self.bottoms)

    @staticmethod
    def margin_of(box: Union['BoundingBox', None]) -> Union[float, None]:
        if box is None:
            return None
        return 2 ** (len(box.tops) - 1) * np.sum(np.subtract(box.tops, box.bottoms))

    @staticmethod
    def volume_of(box: Union['BoundingBox', None]) -> Union[float, None]:
        if box is None:
            return None
        return box.volume()

    @staticmethod
    def overlap_sc(box_1: 'BoundingBox', box_2: 'BoundingBox') -> Union['BoundingBox', None]:
        bb_tops = np.minimum(box_1.tops, box_2.tops)
        bb_bottoms = np.maximum(box_1.bottoms, box_2.bottoms)
        if np.any(bb_tops < bb_bottoms):
            return None
        return BoundingBox(bb_tops, bb_bottoms)

    @staticmethod
    def create(bounds: List['BoundingBox']) -> 'BoundingBox':
        bb_tops = np.max(np.stack([bound.tops for bound in bounds]), axis=0)
        bb_bottoms = np.min(np.stack([bound.tops for bound in bounds]), axis=0)
        return BoundingBox(bb_tops, bb_bottoms)

    def is_same(self, other: 'BoundingBox') -> bool:
        return np.all(self.tops == other.tops) and np.all(self.bottoms == other.bottoms)

    def asymmetry(self, other: 'BoundingBox', dim: int):
        # assumes this is the new one, other is original
        return 2 * (other.center_along(dim) - self.center_along(dim)) / other.width(dim)

    def center_along(self, dim: int):
        return (self.tops[dim] + self.bottoms[dim]) / 2

    def width(self, dim: int):
        return self.tops[dim] - self.bottoms[dim]

    def center(self) -> float:
        return np.divide(np.add(self.tops, self.bottoms), 2)

    def volume(self) -> float:
        return float(np.prod(np.subtract(self.tops, self.bottoms)))

    def volume_diff(self, other: 'BoundingBox') -> float:
        return other.volume() - self.volume()

    def margin(self) -> float:
        # n dimensional perimeter
        return 2 ** (len(self.tops) - 1) * np.sum(np.subtract(self.tops, self.bottoms))

    def margin_diff(self, other: 'BoundingBox') -> float:
        return self.margin() - other.margin()

    def overlap(self, box: 'BoundingBox') -> Union['BoundingBox', None]:
        bb_tops = np.minimum(self.tops, box.tops)
        bb_bottoms = np.maximum(self.bottoms, box.bottoms)
        if np.any(bb_tops < bb_bottoms):
            return None
        return BoundingBox(bb_tops, bb_bottoms)

    def overlap_margin(self, box: 'BoundingBox') -> float:
        overlap = self.overlap(box)
        if overlap is None:
            return 0.0
        return overlap.margin()

    def overlap_volume(self, box: 'BoundingBox') -> float:
        overlap = self.overlap(box)
        if overlap is None:
            return 0.0
        return overlap.volume()

    def min_bb_with(self, box: Union[None, 'BoundingBox']) -> 'BoundingBox':
        """
        mininimum bounding box containing
        :param box: element to preview adding
        :return: BoundingBox
        """
        if box is None:
            return self

        tops = np.maximum(self.tops, box.tops)
        bottoms = np.minimum(self.bottoms, box.bottoms)
        return BoundingBox(tops, bottoms)

    def encloses(self, box: 'BoundingBox') -> bool:
        # faster than checking if the bounding volume changes
        return np.all(self.bottoms < box.bottoms) and np.all(box.tops < self.tops)


@dataclass
class Point(BoundingBox):
    # idiomatically communicate this is a point
    def __init__(self, point: NDArray):
        self.tops = point
        self.bottoms = point
        self.point = point

    def __repr__(self):
        return "Point: " + self.point.__repr__()[6:-1]

    def __copy__(self) -> 'Point':
        return Point(self.tops)


# noinspection SpellCheckingInspection
class RSNode:
    def __init__(self, parent, lower=5, upper=50, shape=0.5):
        # max is max # of values per node before splitting
        self.__lower = lower
        self.__upper = upper
        self.__shape = shape
        self.parent: 'RSNode' = parent

        self.bounds: Union[BoundingBox, None] = None
        self.children: List[Union['RSNode', BoundingBox]] = []
        self.o_box: Union[BoundingBox, None] = None
        self._success = False

    def __repr__(self):
        return self.bounds.__repr__() + ", leaf: " + str(self.is_leaf()) + ", children: " + str(len(self.children))

    def set_bounds(self, bounds: BoundingBox):
        self.bounds = bounds
        return self

    def is_leaf(self):
        if not self.children:
            return True
        if isinstance(self.children[0], BoundingBox):
            return True
        return False

    def choose_subtree(self, element: Union[BoundingBox, Point]) -> 'RSNode':
        # compute covered, (checking if any entries fully cover the object)
        # if there is anything that does, choose the one with minimum volume or perimeter
        covering_nodes: List[RSNode] = []
        for node in self.children:
            if node.bounds.encloses(element):
                covering_nodes += [node]
        if covering_nodes:
            idx = np.argmin(list(map(
                lambda covering_node: covering_node.bounds.volume(),
                covering_nodes
            )))
            return covering_nodes[idx]

        # bounding node must be enlarged

        # otherwise, find growth of overlap's perimeter for each entry if it included point
        # sort in ascending order of the amount each would grow in perimeter
        E: List[RSNode] = [child for child in self.children]
        perim_change = [node.bounds.min_bb_with(element).margin_diff(node.bounds) for node in E]
        E_sorted: List[Tuple[float, RSNode]] = sorted(zip(perim_change, E), key=lambda tup: tup[0], reverse=False)
        E: NDArray[Union[float, RSNode]] = np.array([[x, y] for x, y in E_sorted])
        perim_change, E = (E[:, 0].tolist(), E[:, 1].tolist())

        # assign to first in list if none of the other entries would suffer from assignment
        # (by increased overlap with first node)
        # TODO find way to maintain sorting of this when adding nodes
        node = E[0]
        bb = node.bounds
        bb_with_obj = bb.min_bb_with(element)

        all_valid = True
        last_idx = 0  # for next section
        for i, node in enumerate(E[1:]):
            # offset by 1 due to enumeration and
            # offset by 1 due to list slicing
            idx = i + 2
            n_bb = node.bounds
            if n_bb.overlap_margin(bb_with_obj) - n_bb.overlap_margin(bb) > eps:
                all_valid = False
                # seek last idk where increase in overlap is above 0
                last_idx = idx

        if all_valid:
            return E[0]

        # otherwise, attempt optimization choice:
        # first filter out all but first p entries,
        # where Ep is last entry whose overlap would increase if object inserted to E1
        E = E[:last_idx]
        del_overlap = [0 for _ in range(len(E))]
        candidates: List[RSNode] = []
        self._success = False

        # depth first, find index t among 1 to p where assignment of object to Et
        # would not increase overlap with other entries
        def check_comp(elem: BoundingBox, nodes: List[RSNode], t: int, cand: List[RSNode], method: str):
            if nodes[t] not in cand:
                cand += [nodes[t]]
            nod_bb: BoundingBox = nodes[t].bounds
            app_bb: BoundingBox = nodes[t].bounds.min_bb_with(elem)

            for j, comp in enumerate(nodes):
                if j != t:
                    test_bb: BoundingBox = comp.bounds
                    if method == "perim":
                        inc_overlap = app_bb.overlap_margin(test_bb) - nod_bb.overlap_margin(test_bb)
                        del_overlap[t] += inc_overlap
                    else:
                        inc_overlap = app_bb.overlap_volume(test_bb) - nod_bb.overlap_volume(test_bb)
                        del_overlap[t] += inc_overlap
                    if inc_overlap > eps and nodes[j] not in cand:
                        out = check_comp(elem, nodes, j, cand, method)
                        if self._success:
                            return out
            if del_overlap[t] < eps:
                self._success = True
                return nodes[t]

        # during traversal, all indices are added to candidate list. If depth search not successful,
        # use index which has minimum increase in overlap
        if np.any(np.array(list(map(lambda node: node.bounds.min_bb_with(element).volume(), E))) < eps):
            c = check_comp(element, E, 1, candidates, "perim")
        else:
            c = check_comp(element, E, 1, candidates, "vol")

        if self._success:
            return c
        else:
            return E[np.argmin(del_overlap)]

    def update_parent_bounds(self, bounds: Union[BoundingBox, Point]):
        if self.parent is not None:
            if self.parent.bounds is None:
                self.parent.bounds = bounds
            else:
                self.parent.bounds = self.parent.bounds.min_bb_with(bounds)

    def insert(self, element: Union[BoundingBox]) -> BoundingBox:
        # if node is leaf
        # add point to node list and update bounding box
        # check if split needed
        if self.is_leaf():
            if len(self.children) < self.__upper:
                self.children += [element]
                self.bounds = element.min_bb_with(self.bounds)
                # update parent's bounds
                self.update_parent_bounds(element)

            # splitting a node reduces its bounds, but does not
            # make the bounds of parent nodes any smaller
            # split makes sure the parent's bounds are correct
            if len(self.children) == self.__upper:
                self.choose_split()
            return self.bounds
        else:
            # else:
            # insert to child node
            child = self.choose_subtree(element)
            bounds = child.insert(element)
            self.bounds = self.bounds.min_bb_with(bounds)
            self.update_parent_bounds(element)
            if len(self.children) == self.__upper:
                self.choose_split()
            return self.bounds

    def remove(self, element: Union[BoundingBox, Point]) -> bool:
        pass

    def choose_split(self):
        # chooses split composition of current node, assuming it is overcrowded
        # if internal node:
        if not self.is_leaf():
            # consider all dimensions
            pass

        if self.is_leaf():
            # split candidates
            bots = [child.bottoms for child in self.children]
            tops = [child.tops for child in self.children]

            for dim in range(len(self.bounds.tops.shape)):
                self._minimize_on(dim)

        pass

    def _minimize_on(self, dim):
        max_perim = self.bounds.margin() * 2 - np.min(self.bounds.bottoms)

        top_bbs = sorted(self.children, key=lambda node: node.tops[dim])
        bot_bbs = sorted(self.children, key=lambda node: node.bottoms[dim])
        sc_i: NDArray[BoundingBox] = np.split(
            np.stack([top_bbs, bot_bbs], axis=0),
            np.arange(self.__lower, self.__upper - self.__lower), axis=1)

        sc = np.apply_along_axis(self.__create_sc, sc_i, axes=0)
        wf = self._compute_wf(dim, sc)
        # dim 0: different split indexes
        # dim 1: top vs bottom
        # dim 2: sc_1, sc_2
        # outer apply: over different split indexes
        # inner apply: over different bounding box positions
        # margin of bounding box pairs sc_1 and sc_2
        margin_sc = np.apply_along_axis(lambda block:
                                        np.sum(
                                            np.apply_along_axis(
                                                lambda bbox:
                                                bbox.margin(),
                                                axis=0, arr=block),
                                            axis=0),
                                        axis=0, arr=sc)

        # overlap of bounding box pairs sc_1 and sc_2
        overlap_sc = np.apply_along_axis(lambda block:
                                         np.apply_along_axis(
                                             lambda pair:
                                             BoundingBox.overlap_sc(pair[0], pair[1]),
                                             axis=0, arr=block),
                                         axis=0, arr=sc)

        # margin of overlap of box pairs
        margin_overlap_sc = np.apply_along_axis(lambda vector:
                                                np.apply_along_axis(
                                                    lambda bbox:
                                                    bbox.margin(),
                                                    axis=0, arr=vector),
                                                axis=0, arr=overlap_sc)
        wg: NDArray = np.multiply(margin_sc - max_perim, wf)
        wg_alt: NDArray = np.divide(margin_overlap_sc, wf)

        indexes = np.abs(margin_overlap_sc) < eps
        wg = np.put(wg, indexes, wg_alt[indexes])
        # should give the
        return sc[np.unravel_index(np.argmin(wg), wg.shape)]

    def _compute_wf(self, dim: int, sc: NDArray[BoundingBox]):
        # dim 0: split candidate index
        # dim 1: top vs bottom
        # dim 2: sc_1, sc_2
        asym = np.apply_along_axis(
            lambda split:
            np.apply_along_axis(
                lambda side:
                np.apply_along_axis(
                    lambda box:
                    self.bounds.asymmetry(box, dim),
                    axis=0, arr=side),
                axis=0, arr=split),
            axis=0, arr=sc)

        mean = (1 - self.__lower/(self.__upper + 1)) * asym
        sigma = self.__shape * (1 + np.abs(mean))
        # y offset
        y1 = math.exp(-1/(self.__shape**2))
        # y scaling
        ys = 1/(1 - y1)
        xi = 2*np.arange(self.__lower, self.__upper-self.__lower+1)/(self.__upper+1) - 1

        z_score = (xi - mean)/sigma
        wf = ys*(np.exp(-1*(z_score**2)) - y1)
        return wf

    @staticmethod
    def __create_sc(arr: NDArray) -> NDArray[BoundingBox]:
        return np.array([
            [BoundingBox.create(arr[0][0, :]),
             BoundingBox.create(arr[1][0, :])],
            [BoundingBox.create(arr[0][1, :]),
             BoundingBox.create(arr[1][1, :])]
        ])

    def split(self, group1: List[BoundingBox], group2: List[BoundingBox]):
        pass

    def overflow_treatment(self):
        # if this is the root level *or* it's the first call in this node
        # split node, creating new node as child of parent node.
        pass

    def merge(self):
        # call insert on parent with this node's children
        # parent is self.parent
        if self.parent is None:
            self.parent = RSNode(None, self.__lower, self.__upper, self.__shape)
        for node in self.children:
            self.parent.insert(node)

    def bb_without(self, point: Point):
        pass


class RStarTree:
    def __init__(self, lower=4, upper=50):
        self.__lower = lower
        self.__upper = upper
        # dont forget experimental parameters
        self.root = RSNode(None, lower=lower, upper=upper)
        # root has at least two children unless it is leaf
        # non-leaf nodes have between lower and upper # of children unless root
        # all leaf nodes have between m and M entries unless root
        # all leaves at same depth.

    def __repr__(self):
        return self.root.__repr__()

    def insert(self, element: Union[BoundingBox, Point]):
        bbox = self.root.insert(element)

    def overflow_treatment(self):
        pass

    def reinsert(self):
        pass


if __name__ == "__main__":
    lower = 1
    upper = 4
    rtree = RStarTree(lower=lower, upper=upper)
    cnode_1 = RSNode(rtree.root, lower=lower, upper=upper)
    cnode_1.insert(Point(np.array([3, 5])))
    cnode_2 = RSNode(rtree.root, lower=lower, upper=upper)
    cnode_2.insert(Point(np.array([9, 2])))
    rtree.root.children = [cnode_1, cnode_2]

    rtree.insert(Point(np.array([1, 2])))
    rtree.insert(Point(np.array([2, 3])))
    rtree.insert(Point(np.array([4, 1])))
    rtree.insert(Point(np.array([3, 9])))
    rtree.insert(Point(np.array([4, 7])))
    rtree.insert(Point(np.array([6, 3])))
    rtree.insert(Point(np.array([9, 5])))
    rtree.insert(Point(np.array([5, 4])))
    rtree.insert(Point(np.array([3, 5])))
