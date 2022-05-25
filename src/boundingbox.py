"""
Rights to usage of this implementation and derivative works mine.
Non-commercial rights granted, but works based on this may not
deviate from a license at least as restrictive as this one
"""

from dataclasses import dataclass
from typing import Union, List

import numpy as np
from numpy.typing import NDArray


eps = 0.0001


@dataclass
class BoundingBox:
    # points are abstracted as same top and bottom bounds
    tops: NDArray[float]
    bottoms: NDArray[float]
    obj = None

    def __repr__(self):
        return "BBox(top(" + self.tops.__repr__()[6:-1] \
               + "), bot(" + self.bottoms.__repr__()[6:-1] + "))"

    def __copy__(self) -> 'BoundingBox':
        return BoundingBox(self.tops, self.bottoms)

    def __eq__(self, other: 'BoundingBox'):
        tops = np.abs(np.subtract(self.tops, other.tops))
        bottoms = np.abs(np.subtract(self.bottoms, other.bottoms))

        return np.all(tops < eps) and np.all(bottoms < eps)

    @staticmethod
    def create(bounds: List['BoundingBox']) -> 'BoundingBox':
        bb_tops = np.max(np.stack([bound.tops for bound in bounds]), axis=0)
        bb_bottoms = np.min(np.stack([bound.bottoms for bound in bounds]), axis=0)
        return BoundingBox(bb_tops, bb_bottoms)

    @staticmethod
    def margin_of(box: Union['BoundingBox', None]) -> float:
        if box is None:
            return 0
        return 2 ** (len(box.tops) - 1) * np.sum(np.subtract(box.tops, box.bottoms))

    @staticmethod
    def overlap_sc(box_1: 'BoundingBox', box_2: 'BoundingBox') -> \
            Union['BoundingBox', None]:
        bb_tops = np.minimum(box_1.tops, box_2.tops)
        bb_bottoms = np.maximum(box_1.bottoms, box_2.bottoms)
        if np.any(bb_tops < bb_bottoms):
            return None
        return BoundingBox(bb_tops, bb_bottoms)

    @staticmethod
    def volume_of(box: Union['BoundingBox', None]) -> float:
        if box is None:
            return 0
        return box.volume

    @property
    def center(self) -> float:
        return np.divide(np.add(self.tops, self.bottoms), 2)

    @property
    def margin(self) -> float:
        # n dimensional perimeter
        return 2 ** (len(self.tops) - 1) * np.sum(np.subtract(self.tops, self.bottoms))

    @property
    def volume(self) -> float:
        return float(np.prod(np.subtract(self.tops, self.bottoms)))

    def is_same(self, other: 'BoundingBox') -> bool:
        return np.all(self.tops == other.tops) and np.all(self.bottoms == other.bottoms)

    def asymmetry(self, other: 'BoundingBox', dim: int) -> float:
        # assumes this is the new one, other is original
        other_center = other.center_along(dim)
        self_center = self.center_along(dim)
        return 2 * (self_center - other_center) / max(.5, self.width(dim))

    def center_along(self, dim: int) -> NDArray[float]:
        return (self.tops[dim] + self.bottoms[dim]) / 2

    def width(self, dim: int) -> float:
        return abs(self.tops[dim] - self.bottoms[dim])

    def volume_diff(self, other: 'BoundingBox') -> float:
        return other.volume - self.volume

    def margin_diff(self, other: 'BoundingBox') -> float:
        return self.margin - self.margin_of(other)

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
        return overlap.margin

    def overlap_volume(self, box: 'BoundingBox') -> float:
        overlap = self.overlap(box)
        if overlap is None:
            return 0.0
        return overlap.volume

    def min_bb_with(self, box: Union[None, 'BoundingBox']) -> 'BoundingBox':
        """
        minimum bounding box containing
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
    def __init__(self, point: NDArray[float]):
        self.tops = point
        self.bottoms = point
        self.point = point

    def __repr__(self):
        return "Point(" + self.point.__repr__()[6:-1] + ")"

    def __copy__(self) -> 'Point':
        return Point(self.tops)
