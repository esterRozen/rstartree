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
        return other.margin() - self.margin()

    def overlap(self, box: 'BoundingBox') -> Union['BoundingBox', None]:
        bb_tops = np.min(self.tops, box.tops)
        bb_bottoms = np.max(self.bottoms, box.bottoms)
        if np.any(bb_tops < bb_bottoms):
            return None
        return BoundingBox(bb_tops, bb_bottoms)

    def overlap_margin(self, box: 'BoundingBox') -> float:
        return self.overlap(box).margin()

    def overlap_volume(self, box: 'BoundingBox') -> float:
        return self.overlap(box).volume()

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

    def __copy__(self) -> 'Point':
        return Point(self.tops)


class RSNode:
    def __init__(self, parent=None, lower=5, upper=50):
        # max is max # of values per node before splitting
        self.__lower = lower
        self.__upper = upper
        self.parent: 'RSNode' = parent
        self.bounds: Union[BoundingBox, None] = None
        self.children: List[Union['RSNode', BoundingBox]] = []

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
        for box in self.children:
            if box.bounds.encloses(element):
                covering_nodes += box
        if not covering_nodes:
            idx = np.argmin(list(map(
                lambda node: node.bounds.volume(),
                covering_nodes
            )))
            return covering_nodes[idx]

        # bounding box must be enlarged

        # otherwise, find growth of overlap's perimeter for each entry if it included point
        # sort in ascending order of the amount each would grow in perimeter
        E: List[RSNode] = [child for child in self.children]
        perim_change = [node.bounds.min_bb_with(element).margin_diff(node.bounds) for node in E]
        E_sorted: List[Tuple[float, RSNode]] = sorted(zip(perim_change, E), reverse=False)
        E: NDArray[Union[float, RSNode]] = np.array([[x, y] for x, y in E_sorted])
        perim_change, E = (E[:, 0].tolist(), E[:, 1].tolist())

        # assign to first in list if none of the other entries would suffer from assignment
        # (by increased overlap with first box)
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
        del_overlap = [0 for _ in range(E)]
        candidates: List[RSNode] = []
        success = False

        # depth first, find index t among 1 to p where assignment of object to Et
        # would not increase overlap with other entries
        def check_comp(elem: BoundingBox, nodes: List[RSNode], t: int, cand: List[RSNode], success: bool, method: str):
            if nodes[t] not in cand:
                cand += nodes[t]
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
                        out = check_comp(elem, nodes, j, cand, success, method)
                        if success:
                            return out
            if del_overlap[t] < eps:
                success = True
                return nodes[t]

        # during traversal, all indices are added to candidate list. If depth search not successful,
        # use index which has minimum increase in overlap
        if np.any(list(map(lambda node: node.bounds.min_bb_with(element).volume(), E)) < eps):
            c = check_comp(element, E, 1, candidates, success, "perim")
        else:
            c = check_comp(element, E, 1, candidates, success, "vol")

        if success:
            return c
        else:
            return E[np.argmin(del_overlap)]

    def update_parent_bounds(self, bounds: Union[BoundingBox, Point]):
        if self.parent is not None:
            self.parent.bounds = self.parent.bounds.min_bb_with(bounds)

    def insert(self, element: Union[BoundingBox, Point]) -> BoundingBox:
        # if node is leaf (and element is Point)
        # add point to node list and update bounding box
        # check if split needed
        if self.is_leaf():
            # append element!!
            self.children += [element]
            self.bounds = element.min_bb_with(self.bounds)
            # update parent's bounds
            self.update_parent_bounds(element)

            # splitting a node reduces its bounds, but does not
            # make the bounds of parent nodes any smaller
            # split makes sure the parent's bounds are correct
            if len(self.children) > self.__upper:
                self.choose_split()
            return self.bounds
        else:
            # else:
            # if element is point, insert to child node
            child = self.choose_subtree(element)
            if isinstance(element, Point):
                bounds = child.insert(element)
                self.bounds = self.bounds.min_bb_with(bounds)
                self.update_parent_bounds(element)
                return self.bounds

            # else if node child is leaf
            # this must be bounding box, add to current position
            elif child.is_leaf():
                # append element!!
                self.children += [element]
                self.bounds = self.bounds.min_bb_with(element)
                self.update_parent_bounds(element)
                if len(self.children) > self.__upper:
                    self.choose_split()
                return self.bounds

            # interior node, expand down to children
            else:
                child.insert(element)
                self.update_parent_bounds(element)

    def remove(self, element: Union[BoundingBox, Point]) -> bool:
        pass

    def choose_split(self):
        # splits current node, presuming it is overcrowded
        if self.is_leaf():
            # split candidates
            bots = [child.bottoms for child in self.children]
            tops = [child.tops for child in self.children]

            for dim in range(len(self.bounds.tops.shape)):
                bot_sort_bbs = sorted(self.children, key=lambda child: child.bottoms[dim])
                top_sort_bbs = sorted(self.children, key=lambda child: child.tops[dim])

                max_perim = self.bounds.margin() * 2 - np.min(self.bounds.bottoms)

                margin_splits: List[Tuple[int, float, float]] = []
                ovlp_splits: List[Tuple[int, float, float]] = []
                bbox_splits: List[Tuple[BoundingBox, BoundingBox, BoundingBox, BoundingBox]] = []
                for i in range(self.__lower, self.__upper):
                    sc_i: List[NDArray[BoundingBox]] = np.split(np.stack([top_sort_bbs, bot_sort_bbs], axis=0), [i],
                                                                axis=1)

                    sc_top_1 = BoundingBox.create(sc_i[0][0, :])
                    sc_top_2 = BoundingBox.create(sc_i[1][0, :])
                    sc_bot_1 = BoundingBox.create(sc_i[0][1, :])
                    sc_bot_2 = BoundingBox.create(sc_i[1][1, :])
                    bbox_splits += [(sc_top_1, sc_top_2, sc_bot_1, sc_bot_2)]

                    margin_top = sc_top_1.margin() + sc_top_2.margin()
                    margin_bot = sc_bot_1.margin() + sc_bot_2.margin()
                    margin_splits += [(i, margin_top, margin_bot)]

                    # if this is not a leaf node (it is not),
                    # skip finding lowest margin axis and consider all nodes

                    ovlp_top = BoundingBox.overlap_sc(sc_top_1, sc_top_2)
                    ovlp_bot = BoundingBox.overlap_sc(sc_bot_1, sc_bot_2)

                    ovlp_marg_top = ovlp_top.margin()
                    # ovlp_vol_top = ovlp_top.volume()
                    ovlp_marg_bot = ovlp_bot.margin()
                    # ovlp_vol_bot = ovlp_bot.volume()

                    ovlp_splits += [(i, ovlp_marg_top, ovlp_marg_bot)]
                if np.any(list(map(lambda tup: tup[1] < eps or tup[2] < eps, margin_splits))):
                    # there are overlap free candidates.
                    # skip work and just choose minimum perimeter
                    pass
        pass

    def split(self, group1: List[BoundingBox], group2: List[BoundingBox]):
        pass

    def merge(self, node1: 'RSNode', node2: 'RSNode'):
        # must be siblings
        pass

    def bb_without(self, point: Point):
        pass


class RStarTree:
    def __init__(self, lower=4, upper=50):
        self.__lower = lower
        self.__upper = upper
        # dont forget experimental parameters
        self.root = RSNode(lower=lower, upper=upper)
        # root has at least two children unless it is leaf
        # non-leaf nodes have between lower and upper # of children unless root
        # all leaf nodes have between m and M entries unless root
        # all leaves at same depth.

    def insert(self, element: Union[BoundingBox, Point]):
        self.root.insert(element)

    def overflow_treatment(self):
        pass

    def reinsert(self):
        pass


if __name__ == "__main__":
    rtree = RStarTree(lower=1, upper=4)
    cnode_1 = RSNode(rtree.root)
    cnode_1.insert(Point(np.array([3, 5])))
    cnode_2 = RSNode(rtree.root)
    cnode_2.insert(Point(np.array([9, 2])))
    rtree.root.children = [cnode_1, cnode_2]

    rtree.insert(Point(np.array([1, 2])))
    rtree.insert(Point(np.array([2, 3])))
    rtree.insert(Point(np.array([4, 1])))
    rtree.insert(Point(np.array([3, 9])))
    rtree.insert(Point(np.array([4, 7])))
    rtree.insert(Point(np.array([6, 3])))