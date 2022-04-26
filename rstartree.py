#! /usr/bin/env python3
"""
Rights to usage of this implementation and derivative works mine.
Algorithm based on R* Tree improved method:
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.367.7273&rep=rep1&type=pdf
as well as original algorithm:
https://infolab.usc.edu/csci599/Fall2001/paper/rstar-tree.pdf
"""
from typing import List, Union
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


eps = 0.001


@dataclass
class BoundingBox:
    # points are abstracted as same _tops and _bottoms
    _tops: NDArray[float]
    _bottoms: NDArray[float]

    def __copy__(self) -> 'BoundingBox':
        return BoundingBox(self._tops, self._bottoms)

    def is_same(self, other: 'BoundingBox') -> bool:
        return np.all(self._tops == other._tops) and np.all(self._bottoms == other._bottoms)

    def center(self) -> float:
        return np.divide(np.add(self._tops, self._bottoms), 2)

    def volume(self) -> float:
        return np.prod(np.subtract(self._tops, self._bottoms))

    def volume_diff(self, other: 'BoundingBox') -> float:
        return other.volume() - self.volume()

    def margin(self) -> float:
        # n dimensional perimeter
        return 2 ** (len(self._tops) - 1) * np.sum(np.subtract(np.tops, np.bottoms))

    def margin_diff(self, other: 'BoundingBox') -> float:
        return other.margin()-self.margin()

    def overlap(self, box: 'BoundingBox') -> 'BoundingBox':
        bb_tops = np.min(self._tops, box._tops)
        bb_bottoms = np.max(self._bottoms, box._bottoms)
        return BoundingBox(bb_tops, bb_bottoms)

    def overlap_margin(self, box: 'BoundingBox') -> float:
        return self.overlap(box).margin()

    def overlap_volume(self, box: 'BoundingBox') -> float:
        return self.overlap(box).volume()

    def min_bb_with(self, box: 'BoundingBox') -> 'BoundingBox':
        """
        mininimum bounding box containing
        :param box: element to preview adding
        :return: BoundingBox
        """
        tops = np.maximum(self._tops, box._tops)
        bottoms = np.minimum(self._bottoms, box._tops)
        return BoundingBox(tops, bottoms)

    def encloses(self, box: 'BoundingBox') -> bool:
        # faster than checking if the bounding volume changes
        return np.all(self._bottoms < box._bottoms) and np.all(box._tops < self._tops)


@dataclass
class Point(BoundingBox):
    # idiomatically communicate this is a point
    def __init__(self, point: NDArray):
        self._tops = point
        self._bottoms = point
        self.point = point

    def __copy__(self) -> 'Point':
        return Point(self._tops)


class RSNode:
    def __init__(self, parent=None, lower=5, upper=50):
        # max is max # of values per node before splitting
        self.__lower = lower
        self.__upper = upper
        self.parent = parent
        self.bounds: BoundingBox = None
        self.children: List[Union['RSNode', BoundingBox]] = []

    def set_bounds(self, bounds: BoundingBox):
        self.bounds = bounds
        return self

    def is_leaf(self):
        if self.children is None:
            return True
        if isinstance(self.children[0], BoundingBox):
            return True
        return False

    def choose_subtree(self, element: BoundingBox) -> 'RSNode':
        # compute covered, (checking if any entries fully cover the object)
        # if there is anything that does, choose the one with minimum volume or perimeter
        covering_nodes: List[RSNode] = []
        for box in self.children:
            if box.bounds.encloses(element):
                covering_nodes += box
        if covering_nodes is not None:
            idx = np.argmin(list(map(
                lambda node: node.bounds.volume(),
                covering_nodes
            )))
            return covering_nodes[idx]

        # bounding box must be enlarged

        # otherwise, find growth of overlap's perimeter for each entry if it included point
        # sort in ascending order of the amount each would grow in perimeter
        E: List[RSNode] = [child.bounds for child in self.children]
        perim_change = [node.bounds.min_bb_with(element).margin_diff(node.bounds) for node in E]
        E: NDArray[RSNode] = np.array([[x, y] for x, y in sorted(zip(perim_change, E), reverse=False)])
        perim_change, E = (E[:, 0].tolist(), E[:, 1].tolist())

        # assign to first in list if none of the other entries would suffer from assignment
        # (by increased overlap with first box)
        # TODO find way to maintain sorting of this when adding nodes
        node = E[0]
        bb = node.bounds
        bb_with_obj = bb.min_bb_with(element)

        all_valid = True
        last_idx = 0     # for next section
        for i, node in enumerate(E[1:]):
            # offset by 1 due to enumeration and
            # offset by 1 due to list slicing
            idx = i+2
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

        # depth first, find index t among 1..p where assignment of object to Et
        # would not increase overlap with other entries
        def check_comp(object: BoundingBox, nodes: List[RSNode], t: int, cand: List[RSNode], success: bool, method: str):
            if nodes[t] not in cand:
                cand += nodes[t]
            nod_bb: BoundingBox = nodes[t].bounds
            app_bb: BoundingBox = nodes[t].bounds.min_bb_with(object)

            out = None
            for j, comp in enumerate(nodes):
                if j != t:
                    test_bb: BoundingBox = comp.bounds
                    if method == "perim":
                        inc_overlap = app_bb.overlap_margin(comp.bounds) - nod_bb.overlap_margin(comp.bounds)
                        del_overlap[t] += inc_overlap
                    else:
                        inc_overlap = app_bb.overlap_volume(comp.bounds) - nod_bb.overlap_volume(comp.bounds)
                        del_overlap[t] += inc_overlap
                    if inc_overlap > eps and nodes[j] not in cand:
                        out = check_comp(object, nodes, j, cand, success, method)
                        if success:
                            return out
            if del_overlap[t] < eps:
                success = True
                return nodes[t]

        # during traveral, all indices are added to candidate list. If depth search not succesful,
        # use index which has minimum increase in overlap
        if np.any(list(map(lambda node: node.bounds.min_bb_with(element).volume(), E)) < eps):
            c = check_comp(element, E, 1, candidates, success, "perim")
        else:
            c = check_comp(element, E, 1, candidates, success, "vol")

        if success:
            return c
        else:
            return E[np.argmin(del_overlap)]

    def insert(self, element: Union[BoundingBox, Point]) -> BoundingBox:
        # if node is leaf (and element is Point)
        # add point to node list and update bounding box
        # check if split needed
        if self.is_leaf():
            # append element!!
            self.children += element
            self.bounds = self.bounds.min_bb_with(element)
            if len(self.children) > self.__upper:
                self.split()
            # have to pass bounds up to update higher nodes
            return self.bounds

        else:
            # else:
            # if element is point, insert to child node
            child = self.choose_subtree(element)
            if isinstance(element, Point):
                bounds = child.insert(element)
                self.bounds = self.bounds.min_bb_with(bounds)
                return self.bounds

            # else if node child is leaf
            # this must be bounding box, add to current position
            elif child.is_leaf():
                # append element!!
                self.children += element
                self.bounds = self.bounds.min_bb_with(element)
                if len(self.children) > self.__upper:
                    self.split()
                return self.bounds

            # interior node, expand down to children
            else:
                bounds = child.insert(element)
                self.bounds = self.bounds.min_bb_with(bounds)
                return self.bounds

    def split(self):
        # splits current node
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

    def insert(self, object: BoundingBox):
        pass

    def overflow_treatment(self):
        pass

    def reinsert(self):
        pass

    def split(self, node: RSNode):

        pass
