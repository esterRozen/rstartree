#! /usr/bin/env python3
"""
Rights to usage of this implementation and derivative works mine.
Algorithm based on R* Tree improved method:
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.367.7273&rep=rep1&type=pdf
as well as original algorithm:
https://infolab.usc.edu/csci599/Fall2001/paper/rstar-tree.pdf
"""
from typing import List
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class BoundingBox:
    # points are abstracted as same tops and bottoms
    tops: NDArray[float]
    bottoms: NDArray[float]

    def center(self) -> float:
        return np.divide(np.add(self.tops, self.bottoms), 2)

    def volume(self) -> float:
        return np.prod(np.subtract(self.tops, self.bottoms))

    def volume_diff(self, other: 'BoundingBox') -> float:
        return other.volume() - self.volume()

    def margin(self) -> float:
        # n dimensional perimeter
        return 2**(len(self.tops)-1) * np.sum(np.subtract(np.tops, np.bottoms))

    def margin_diff(self, other: 'BoundingBox') -> float:
        return other.margin()-self.margin()

    def overlap(self, box: 'BoundingBox') -> 'BoundingBox':
        bb_tops = np.min(self.tops, box.tops)
        bb_bottoms = np.max(self.bottoms, box.bottoms)
        return BoundingBox(bb_tops, bb_bottoms)

    def overlap_margin(self, box: 'BoundingBox') -> float:
        return self.overlap(box).margin()

    def overlap_volume(self, box: 'BoundingBox') -> float:
        return self.overlap(box).volume()

    def is_same(self, other) -> bool:
        return np.all(self.tops == other.tops) and np.all(self.bottoms == other.bottoms)

    def min_bb_with(self, box: 'BoundingBox') -> 'BoundingBox':
        tops = np.maximum(self.tops, box.tops)
        bottoms = np.minimum(self.bottoms, box.tops)
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


class RSNode:
    def __init__(self, parent=None):
        self.parent = parent
        self.bounds: BoundingBox = None
        self.children: List[RSNode.__class__()] = []
        self.values: List[Point] = []

    def set_bounds(self, bounds: BoundingBox):
        self.bounds = bounds
        return self

    def insert(self, object: BoundingBox):
        pass


class RStarTree:
    def __init__(self):
        # dont forget experimental parameters
        self.root = None

    def choose_subtree(self, node: RSNode, object: BoundingBox):
        # compute covered, (checking if any entries fully cover the object)
        # if there is anything that does, choose the one with minimum volume or perimeter
        covering_nodes: List[RSNode] = []
        for box in node.children:
            if box.bounds.encloses(object):
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
        E: List[RSNode] = [child.bounds for child in node.children]
        perim_change = [node.bounds.min_bb_with(object).margin_diff(node.bounds) for node in E]
        E: NDArray[RSNode] = np.array([[x, y] for x, y in sorted(zip(perim_change, E), reverse=False)])
        perim_change, E = (E[:, 0].tolist(), E[:, 1].tolist())

        # assign to first in list if none of the other entries would suffer from assignment
        # (by increased overlap with first box)
        # TODO find way to maintain sorting of this when adding nodes
        node = E[0]
        bb = node.bounds
        bb_with_obj = bb.min_bb_with(object)

        all_valid = True
        last_idx = 0     # for next section
        for i, node in enumerate(E[1:]):
            # offset by 1 due to enumeration and
            # offset by 1 due to list slicing
            idx = i+2
            n_bb = node.bounds
            if n_bb.overlap_margin(bb_with_obj) - n_bb.overlap_margin(bb) > 0.0001:
                all_valid = False
                # seek last idk where increase in overlap is above 0
                last_idx = idx

        if all_valid:
            return E[0]

        # otherwise, attempt optimization choice:
        # first filter out all but first p entries,
        # where Ep is last entry whose overlap would increase if object inserted to E1
        E = E[:last_idx]
        candidates: List[RSNode] = []
        success = False


        # depth first, find index t among 1..p where assignment of object to Et
        # would not increase overlap with other entries

        # during traveral, all indices added to candidate list. If depth search not succesful,
        # use index which has minimum increase in overlap
        pass

    def insert(self, object: BoundingBox):
        pass

    def overflow_treatment(self):
        pass

    def reinsert(self):
        pass

    def split(self):
        pass
