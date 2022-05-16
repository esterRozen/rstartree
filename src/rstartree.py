"""
Rights to usage of this implementation and derivative works mine.
Non-commercial rights granted, but works based on this may not
deviate from a license at least as restrictive as this one

Algorithm based on R* Tree improved method:
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.367.7273&rep=rep1&type=pdf
as well as original algorithm:
https://infolab.usc.edu/csci599/Fall2001/paper/rstar-tree.pdf
some r* tree algos which weren't majorly different
from their r tree counterpart loosely based on:
https://www.mathcs.emory.edu/~cheung/Courses/554/Syllabus/3-index/R-tree3.html
"""
import math
from typing import List, Union, Tuple
import numpy as np
from numpy.typing import NDArray

from .boundingbox import BoundingBox, Point

eps = 0.001


# noinspection SpellCheckingInspection
class RSNode:
    def __init__(self, parent, tree: 'RStarTree', shape=0.5):
        # can access global parameters
        self.__tree = tree
        self.parent: 'RSNode' = parent

        self.bounds: Union[BoundingBox, None] = None
        self.children: List[Union['RSNode', BoundingBox]] = []
        self.o_box: Union[BoundingBox, None] = None
        # TODO remove this
        self._success = False

    def __repr__(self):
        return self.bounds.__repr__() + ", leaf: " \
               + str(self.is_leaf) + ", children: " + str(len(self.children))

    def _choose_subtree(self, element: Union[BoundingBox, Point]) -> 'RSNode':
        # compute covered, (checking if any entries fully cover the object)
        # if there is anything that does, choose the one with minimum volume or perimeter
        covering_nodes: List[RSNode] = []
        for node in self.children:
            if node.bounds.encloses(element):
                covering_nodes += [node]

        # if any in covering nodes
        if covering_nodes:
            idx = np.argmin(list(map(
                    lambda covering_node: covering_node.bounds.volume,
                    covering_nodes
            )))
            return covering_nodes[idx]

        # bounding node must be enlarged

        # otherwise, find growth of overlap's perimeter for each entry if it included point
        # sort in ascending order of the amount each would grow in perimeter
        E: List[RSNode] = [child for child in self.children]
        perim_change = [node.bounds.min_bb_with(element).margin_diff(node.bounds) for node in E]
        E_sorted: List[Tuple[float, RSNode]] = sorted(
                zip(perim_change, E),
                key=lambda tup: tup[0], reverse=False)
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
        for i, remaining in enumerate(E[1:]):
            # offset by 1 due to enumeration and
            # offset by 1 due to list slicing
            idx = i + 2
            n_bb = remaining.bounds
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
        if np.any(np.array(list(map(lambda cand: cand.bounds.min_bb_with(element).volume, E))) < eps):
            c = check_comp(element, E, 1, candidates, "perim")
        else:
            c = check_comp(element, E, 1, candidates, "vol")

        if self._success:
            return c
        else:
            return E[np.argmin(del_overlap)]

    ##########################
    # choose subtree functions.
    def __check_coverage(self):
        pass

    def __find_growth_overlap_perimeter(self):
        pass

    def __check_node_one(self):
        pass

    def __consider_candidates(self):
        pass

    def __best_candidate(self):
        pass

    ##########################

    def query(self, element: BoundingBox) -> bool:
        # TODO query
        pass

    def insert(self, element: Union[BoundingBox, Point]) -> None:
        """
        inserts using heuristic function to choose path
        to insert down
        :param element: BBox or Point to be inserted in nearly
        optimal location
        """

        # if node is leaf
        # add point to node list and update bounding box
        # check if split needed
        if self.is_leaf:
            if self.is_overfilled:
                self._split()
                self.parent.insert(element)
                return
            else:
                self.children += [element]

                # bounds of element *with* own bounds deals
                # with issue where self.bounds is None
                self.bounds = element.min_bb_with(self.bounds)
                return

        else:
            # else:
            # insert to child node
            if self.is_overfilled:
                self._split()
                self.parent.insert(element)
                return
            else:
                child = self._choose_subtree(element)
                child.insert(element)

                self.bounds = element.min_bb_with(self.bounds)
                return

    def _split(self) -> None:
        """
        splits the current node. if root, will unseat current tree's root
        :return: None
        """
        new_nodes = self._split_in_two()

        new_nodes[0].bounds = BoundingBox.create(
                [child.bounds for child in new_nodes[0].children])
        new_nodes[1].bounds = BoundingBox.create(
                [child.bounds for child in new_nodes[1].children])

        if self.is_root:
            new_root = RSNode(None, self.__tree)

            new_root.bounds = self.bounds

            new_nodes[0].parent = new_root
            new_nodes[1].parent = new_root

            new_root.children = new_nodes

            new_root.__tree.root = new_root
        else:
            self.parent.children.remove(self)

            new_nodes[0].parent = self.parent
            new_nodes[1].parent = self.parent

            self.parent.children += new_nodes

    def _split_in_two(self) -> List['RSNode']:
        """
        chooses split composition of node currently inside,
        assuming it is overcrowded
        :return: 2 generic nodes with children of split node.
        """

        # if internal node:
        if self.is_leaf:
            # TODO choose best among splits
            # only determine likely good split dimension
            dim = self.__determine_dim()
            split = [self.__minimize_on(dim)]
        else:
            # consider all dimensions
            splits = []
            for dim in range(self.bounds.tops.shape[0]):
                splits += [self.__minimize_on(dim)]


        # testing stuff!!
        new_nodes = [RSNode(None, self.__tree), RSNode(None, self.__tree)]
        new_nodes[0].children = self.children[0:2]
        new_nodes[1].children = self.children[2:]
        return new_nodes

    def __determine_dim(self) -> int:
        """
        provides the indexes of dimension with
        the smallest split perimeter possible
        :return: Tuple of dimension and direction (top/bottom, 0/1)
        """

        best = None
        for dim in range(self.bounds.tops.shape[0]):
            top_bbs = sorted(self.children, key=lambda node: node.tops[dim])
            bot_bbs = sorted(self.children, key=lambda node: node.bottoms[dim])

            sc_i: List[List[BoundingBox]] = []
            for idx in range(self.__lower, self.__upper - self.__lower + 1):
                sc = [top_bbs[0:idx], top_bbs[idx:]]
                sc = [BoundingBox.create(sc[0]), BoundingBox.create(sc[1])]
                sc_i += [sc[0].margin + sc[1].margin]

                sc = [bot_bbs[0:idx], bot_bbs[idx:]]
                sc = [BoundingBox.create(sc[0]), BoundingBox.create(sc[1])]
                sc_i += [sc[0].margin + sc[1].margin]

            minimum = np.argmin(sc_i)
            if best is None or sc_i[minimum] < best[1]:
                best = (dim, sc_i[minimum])

        return best[0]

    def __minimize_on(self, dim: int) -> Tuple[int, int, float]:
        """
        provides best split candidate of a node for a given dimension
        :param dim: dimension minimized over
        :return: Tuple displaying: int, top/bottom (0/1), score
        """
        max_perim = self.bounds.margin * 2 - np.min(self.bounds.bottoms)

        top_bbs = sorted(self.children, key=lambda node: node.tops[dim])
        bot_bbs = sorted(self.children, key=lambda node: node.bottoms[dim])
        sc_i: NDArray[BoundingBox] = np.split(
                np.stack([top_bbs, bot_bbs], axis=0),
                np.arange(self.__lower, self.__upper - self.__lower), axis=1)

        sc = np.apply_along_axis(self.__create_sc, sc_i, axes=0)
        wf = self.__compute_wf(dim, sc)
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
                                                        bbox.margin,
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
                                                        bbox.margin,
                                                        axis=0, arr=vector),
                                                axis=0, arr=overlap_sc)
        wg: NDArray = np.multiply(margin_sc - max_perim, wf)
        wg_alt: NDArray = np.divide(margin_overlap_sc, wf)

        indexes = np.abs(margin_overlap_sc) < eps
        wg = np.put(wg, indexes, wg_alt[indexes])
        # should give the
        return sc[np.unravel_index(np.argmin(wg), wg.shape)]

    def __compute_wf(self, dim: int, sc: NDArray[BoundingBox]):
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

        mean = (1 - self.__lower / (self.__upper + 1)) * asym
        sigma = self.__shape * (1 + np.abs(mean))
        # y offset
        y1 = math.exp(-1 / (self.__shape ** 2))
        # y scaling
        ys = 1 / (1 - y1)
        num = 2 * np.arange(self.__lower,
                            self.__upper - self.__lower + 1)
        xi = num / (self.__upper + 1) - 1

        z_score = (xi - mean) / sigma
        wf = ys * (np.exp(-1 * (z_score ** 2)) - y1)
        return wf

    @property
    def __lower(self):
        return self.__tree.lower

    @property
    def __upper(self):
        return self.__tree.upper

    @property
    def __shape(self):
        return self.__tree.shape

    @property
    def is_leaf(self):
        if not self.children:
            return True
        if isinstance(self.children[0], BoundingBox):
            return True
        return False

    @property
    def is_overfilled(self):
        return self.children.__len__() >= self.__tree.upper

    @property
    def is_root(self):
        return self.parent is None

    @staticmethod
    def __create_sc(arr: NDArray) -> NDArray[BoundingBox]:
        return np.array([
            [BoundingBox.create(arr[0][0, :]),
             BoundingBox.create(arr[1][0, :])],
            [BoundingBox.create(arr[0][1, :]),
             BoundingBox.create(arr[1][1, :])]
        ])


class RStarTree:
    def __init__(self, lower=4, upper=50, shape=0.5):
        self.lower = lower
        self.upper = upper
        self.shape = shape
        # dont forget experimental parameters
        self.root = RSNode(None, self)
        # root has at least two children unless it is leaf
        # non-leaf nodes have between lower and upper # of children unless root
        # all leaf nodes have between m and M entries unless root
        # all leaves at same depth.

    def __repr__(self):
        return self.root.__repr__()

    def insert(self, element: Union[BoundingBox, Point]):
        self.root.insert(element)

    # check if present
    def query(self, element: Union[BoundingBox, Point]) -> bool:
        return self.root.query(element)

    def remove(self, element: Union[BoundingBox, Point]) -> bool:
        pass

    def nearest_neighbor(self, element: Union[BoundingBox, Point], k=1) -> \
            List[Union[BoundingBox, Point]]:
        pass


if __name__ == "__main__":
    rtree = RStarTree(lower=1, upper=4)
    cnode_1 = RSNode(rtree.root, rtree)
    cnode_1.insert(Point(np.array([3, 5])))
    cnode_2 = RSNode(rtree.root, rtree)
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
