#! /usr/bin/env python3
from typing import List
from dataclasses import dataclass


@dataclass
class Bounds:
    top: float
    right: float
    bottom: float
    left: float

    def center(self) -> float:
        return ((self.top + self.bottom) / 2, (self.left + self.right) / 2)

    def area(self) -> float:
        return (self.top - self.bottom) * (self.right - self.left)

    def margin(self) -> float:
        return 2 * (self.top - self.bottom + self.right - self.left)

    def overlap(self, other):
        bb_top = min(self.top, other.top)
        bb_right = min(self.right, other.right)
        bb_bottom = max(self.bottom, other.bottom)
        bb_left = max(self.left, other.left)
        return Bounds(bb_top, bb_right, bb_bottom, bb_left).area()


@dataclass
class Point:
    x: float
    y: float


class Node:
    def __init__(self, parent=None):
        self.parent = parent
        self.bounds: Bounds = None
        self.children: List[Any] = []
        self.values: List[Point] = []

    def set_bounds(self, bounds: Bounds):
        self.bounds = bounds
        return self

    def insert(self, point: Point):
        pass


class RStarTree:
    def __init__(self):
        self.root = None
