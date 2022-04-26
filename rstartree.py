#! /usr/bin/env python3
from _typeshed import Self
from typing import List, Any
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class Point:
    val: NDArray


@dataclass
class BoundingBox:
    tops: NDArray[float]
    bottoms: NDArray[float]

    def center(self) -> float:
        return np.divide(np.add(self.tops, self.bottoms), 2)

    def area(self) -> float:
        return np.prod(np.subtract(self.tops, self.bottoms))

    def area_diff(self, other: 'BoundingBox') -> float:
        return other.area()-self.area()

    def margin(self) -> float:
        return 2**(len(self.tops)-1) * np.sum(np.subtract(np.tops, np.bottoms))

    def margin_diff(self, other: 'BoundingBox') -> float:
        return other.margin()-self.margin()

    def overlap(self, other) -> float:
        bb_tops = np.min(self.tops, other.tops)
        bb_bottoms = np.max(self.bottoms, other.bottoms)
        return BoundingBox(bb_tops, bb_bottoms).area()

    def is_same(self, other) -> bool:
        return np.all(self.tops == other.tops) and np.all(self.bottoms == other.bottoms)

    def min_bb_with(self, point: Point) -> Self:
        tops = np.maximum(self.tops, point)
        bottoms = np.minimum(self.bottoms, point)
        return BoundingBox(tops, bottoms)


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
        pass

    def insert(self, point: Point):
        pass

    def overflow_treatment(self):
        pass

    def reinsert(self):
        pass

    def split(self):
        pass
