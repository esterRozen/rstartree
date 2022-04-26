#! /usr/bin/env python3
from typing import List, Any
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

    def overlap(self, box: 'BoundingBox') -> float:
        bb_tops = np.min(self.tops, box.tops)
        bb_bottoms = np.max(self.bottoms, box.bottoms)
        return BoundingBox(bb_tops, bb_bottoms).volume()

    def is_same(self, other) -> bool:
        return np.all(self.tops == other.tops) and np.all(self.bottoms == other.bottoms)

    def min_bb_with(self, box: 'BoundingBox') -> Self:
        tops = np.maximum(self.tops, box.tops)
        bottoms = np.minimum(self.bottoms, box.tops)
        return BoundingBox(tops, bottoms)

    def is_contained(self, box: 'BoundingBox') -> bool:
        # faster than checking if the bounding volume changes
        return np.all(self.bottoms < box.bottoms) and np.all(box.tops < self.tops)


@dataclass
class Point(BoundingBox):
    # idiomatically communicate this is a point
    def __init__(self, point: NDArray):
        self.tops = point
        self.bottoms = point


class Node:
    def __init__(self, parent=None):
        self.parent = parent
        self.bounds: BoundingBox = None
        self.children: List[Any] = []
        self.values: List[Point] = []

    def set_bounds(self, bounds: BoundingBox):
        self.bounds = bounds
        return self

    def insert(self, point: Point):
        pass


class RStarTree:
    def __init__(self):
        # dont forget experimental parameters
        self.root = None

    def choose_subtree(self,):
        # compute covered, (checking if any entries fully cover the object)
        # if there is anything that does, choose the one with minimum volume or perimeter


        # otherwise, find growth of perimeter for each entry if it included point
        # sort in ascending order of the amount each would grow in perimeter

        # assign to first in list if none of the other entries would suffer from assignment
        # (by increased overlap with first box)

        # otherwise, attempt optimization choice:
        # first filter out all but first p entries,
        # where Ep is last entry whose overlap would increase if object inserted to E1

        # depth first, find index t among 1..p where assignment of object to Et
        # would not increase overlap with other entries

        # during traveral, all indices added to candidate list. If depth search not succesful,
        # use index which has minimum increase in overlap
        pass

    def insert(self, point: Point):
        pass

    def overflow_treatment(self):
        pass

    def reinsert(self):
        pass

    def split(self):
        pass
