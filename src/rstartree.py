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
from typing import List, Union, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from .boundingbox import BoundingBox, Point


eps = 0.001


# noinspection SpellCheckingInspection
def create_sc_bounds(bbs_sorted: List[BoundingBox], idx: int) -> \
        List[BoundingBox]:
    sc = [bbs_sorted[:idx], bbs_sorted[idx:]]
    return [BoundingBox.create(sc[0]), BoundingBox.create(sc[1])]


class RSNode:
    def __init__(self, parent, tree: 'RStarTree'):
        # used to access global parameters
        self.__tree = tree
        self.parent: 'RSNode' = parent

        self.bounds: Union[BoundingBox, None] = None
        self.children: List[Union['RSNode', BoundingBox]] = []

        # set at first instantiation (for inner nodes)
        # (or first time a leaf node has over m objects)
        self.o_box: Union[BoundingBox, None] = None
        # TODO remove this. it's terrible
        self._success = False

    def __repr__(self):
        return "Node(" + self.bounds.__repr__() \
               + ", leaf: " + str(self.is_leaf) \
               + ", children: " + str(len(self.children)) \
               + ")"

    @property
    def height(self):
        if self.is_leaf:
            return 1
        else:
            return 1 + self.children[0].height

    @property
    def depth(self):
        if self.is_root:
            return 0
        else:
            return 1 + self.parent.depth

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
    def is_underfilled(self):
        return self.children.__len__() < self.__tree.lower

    @property
    def is_root(self):
        return self.parent is None

    # main interaction methods

    def query(self, element: BoundingBox) -> bool:
        # TODO query (high priority)
        pass

    def insert(self, element: BoundingBox) -> None:
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

                if not self.is_root:
                    self.parent.insert(element)
                else:
                    self.__tree.root.insert(element)
                return

            else:
                # should only happen a few times!
                # (may happen if elements are removed from a node)
                if self.is_underfilled:
                    self.o_box = element.min_bb_with(self.bounds)

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

                if not self.is_root:
                    self.parent.insert(element)
                else:
                    self.__tree.root.insert(element)
                return

            else:
                child = self._choose_subtree(element)
                child.insert(element)

                self.bounds = element.min_bb_with(self.bounds)
                return

    def remove(self, element: BoundingBox) -> bool:
        # TODO remove (low priority)
        #  should have some means of handling recombining nodes
        #  should recursively go up tree if recombined nodes cause
        #  parent node to have too few nodes.
        pass

    ##########################
    # internals
    # choose subtree

    def _choose_subtree(self, element: BoundingBox) -> 'RSNode':
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
    # Split

    def _split(self) -> None:
        """
        splits the current node. if root, will unseat current tree's root
        :return: None
        """
        new_nodes = self._split_in_two()

        if new_nodes[0].is_leaf:
            new_nodes[0].bounds = BoundingBox.create(new_nodes[0].children)
            new_nodes[1].bounds = BoundingBox.create(new_nodes[1].children)
        else:
            new_nodes[0].bounds = BoundingBox.create(
                    [child.bounds for child in new_nodes[0].children])
            new_nodes[1].bounds = BoundingBox.create(
                    [child.bounds for child in new_nodes[1].children])

        new_nodes[0].o_box = new_nodes[0].bounds
        new_nodes[1].o_box = new_nodes[1].bounds

        if self.is_root:
            new_root = RSNode(None, self.__tree)

            new_root.bounds = self.bounds
            new_root.o_box = self.bounds

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
            # consider only one dim for leaf
            dim = self.__determine_dim()
            best = self.__minimize_on(dim)
        else:
            # consider all dimensions
            best: Optional[Tuple[int, int, float]] = None
            for dim in range(self.bounds.tops.shape[0]):
                split = self.__minimize_on(dim)
                if best is None or split[2] < best[2]:
                    best = split

        new_nodes = [RSNode(None, self.__tree), RSNode(None, self.__tree)]

        # check direction
        if best[1]:
            if self.is_leaf:
                node_sort = sorted(self.children, key=lambda bbox: bbox.bottoms[dim])
            else:
                node_sort = sorted(self.children, key=lambda node: node.bounds.bottoms[dim])
        else:
            if self.is_leaf:
                node_sort = sorted(self.children, key=lambda bbox: bbox.tops[dim])
            else:
                node_sort = sorted(self.children, key=lambda node: node.tops[dim])

        new_nodes[0].children = node_sort[:best[0] + self.__lower]
        new_nodes[1].children = node_sort[best[0] + self.__lower:]

        return new_nodes

    def __sort_nodes_over(self, dim: int) -> \
            Tuple[List[BoundingBox], List[BoundingBox]]:
        top_nodes = sorted(self.children, key=lambda node: node.tops[dim])
        bot_nodes = sorted(self.children, key=lambda node: node.bottoms[dim])

        top_bbs = list(map(lambda rnode: rnode.bounds, top_nodes))
        bot_bbs = list(map(lambda rnode: rnode.bounds, bot_nodes))
        return top_bbs, bot_bbs

    def __sort_bounds_over(self, dim: int) -> \
            Tuple[List[BoundingBox], List[BoundingBox]]:
        top_bbs = sorted(self.children, key=lambda bbox: bbox.tops[dim])
        bot_bbs = sorted(self.children, key=lambda bbox: bbox.bottoms[dim])

        return top_bbs, bot_bbs

    def __determine_dim(self) -> int:
        """
        provides the indexes of dimension with
        the smallest split perimeter possible
        :return: Tuple of dimension and direction (top/bottom, 0/1)
        """
        best = None
        for dim in range(self.bounds.tops.shape[0]):
            top_bbs, bot_bbs = self.__sort_bounds_over(dim)

            sc_i: List[List[BoundingBox]] = []
            for idx in range(self.__lower, self.__upper - self.__lower + 1):
                sc = create_sc_bounds(top_bbs, idx)
                sc_i += [sc[0].margin + sc[1].margin]

                sc = create_sc_bounds(bot_bbs, idx)
                sc_i += [sc[0].margin + sc[1].margin]

            minimum = np.argmin(sc_i)
            if best is None or sc_i[minimum] < best[1]:
                best = (dim, sc_i[minimum])

        return best[0]

    def __minimize_on(self, dim: int) -> Tuple[int, int, float]:
        """
        provides best split candidate of a node for a given dimension
        :param dim: dimension minimized over
        :return: Tuple with: idx, top/bottom (0/1), cost
        """

        # always assume you are in the node that is being split!
        max_perim = self.bounds.margin - np.min(self.bounds.bottoms)

        if self.is_leaf:
            top_bbs, bot_bbs = self.__sort_bounds_over(dim)
        else:
            top_bbs, bot_bbs = self.__sort_nodes_over(dim)

        sc_i_list: List[List[List[BoundingBox]]] = []
        for idx in range(self.__lower, self.__upper - self.__lower + 1):
            sc_i_list += [[create_sc_bounds(top_bbs, idx),
                           create_sc_bounds(bot_bbs, idx)]]

        # dim 0: different split indexes
        # dim 1: top vs bottom
        # dim 2: sc_1, sc_2

        wf = self.__compute_wf(dim)

        # outer apply: over different split indexes
        # inner apply: over different bounding box positions
        # margin of bounding box pairs sc_1 and sc_2
        margin_sc: List[List[float]] = []
        for side_list in sc_i_list:
            margin_sc += [[]]
            for split in side_list:
                margin_sc[-1] += [split[0].margin + split[1].margin]

        # TODO if there are ties for best margin, choose by best area

        # overlap of bounding box pairs sc_1 and sc_2
        overlap_sc: List[List[BoundingBox]] = []
        for side_list in sc_i_list:
            overlap_sc += [[]]
            for split in side_list:
                overlap_sc[-1] += [BoundingBox.overlap_sc(split[0], split[1])]

        # margin of overlap of box pairs
        margin_overlap_sc: List[List[float]] = []
        for side_list in overlap_sc:
            margin_ovlp_sc_1 = 0
            margin_ovlp_sc_2 = 0
            if side_list[0] is not None:
                margin_ovlp_sc_1 = side_list[0].margin
            if side_list[1] is not None:
                margin_ovlp_sc_2 = side_list[1].margin
            margin_overlap_sc += [[
                margin_ovlp_sc_1,
                margin_ovlp_sc_2
            ]]

        margin_sc: NDArray = np.array(margin_sc)
        wg: NDArray = np.zeros(margin_sc.shape)
        wg[:, 0] = np.multiply(margin_sc[:, 0] - max_perim, wf)
        wg[:, 1] = np.multiply(margin_sc[:, 1] - max_perim, wf)

        margin_overlap_sc: NDArray = np.array(margin_overlap_sc)
        wg_alt: NDArray = np.zeros(margin_overlap_sc.shape)
        wg_alt[:, 0] = np.divide(margin_overlap_sc[:, 0], wf)
        wg_alt[:, 1] = np.divide(margin_overlap_sc[:, 1], wf)

        indexes = np.abs(margin_overlap_sc) > eps
        np.putmask(wg, indexes, wg_alt[indexes])

        # should give the best split candidate
        split, direction = np.unravel_index(np.argmin(wg), wg.shape)

        return split, direction, wg[split, direction]

    def __compute_wf(self, dim: int) -> List[float]:
        # how much the bbox has become lopsided along
        # dimension dim, relative to where it started
        asym = self.bounds.asymmetry(self.o_box, dim)

        mean = (1 - 2 * self.__lower / (self.__upper + 1)) * asym
        sigma = self.__shape * (1 + np.abs(mean))

        # y offset
        y1 = math.exp(-1 / (self.__shape ** 2))
        # y scaling
        ys = 1 / (1 - y1)
        num = np.arange(self.__lower,
                        self.__upper - self.__lower + 1)
        xi = 2 * num / self.__upper - 1

        z_score = (xi - mean) / sigma
        wf = ys * (np.exp(-1 * (z_score ** 2)) - y1)
        return wf

    # internal properties
    @property
    def __lower(self):
        return self.__tree.lower

    @property
    def __upper(self):
        return self.__tree.upper

    @property
    def __shape(self):
        return self.__tree.shape


class RStarTree:
    def __init__(self, lower=4, upper=50, shape=0.5):
        # helps maintain invariants by preventing these from being changed
        self.__lower = lower
        self.__upper = upper
        self.__shape = shape
        # dont forget experimental parameters
        self.root = RSNode(None, self)
        # root has at least two children unless it is leaf
        # non-leaf nodes have between lower and upper # of children unless root
        # all leaf nodes have between m and M entries unless root
        # all leaves at same depth.
        # (the tree can only grow in depth from the root, which makes it easier)

    def __repr__(self):
        return "Tree: Root(" + self.root.__repr__() + ")"

    # getters
    @property
    def lower(self):
        return self.__lower

    @property
    def upper(self):
        return self.__upper

    @property
    def shape(self):
        return self.__shape

    @property
    def height(self):
        return self.root.height

    def query(self, element: BoundingBox) -> bool:
        # check if present
        return self.root.query(element)

    # setters
    def insert(self, element: BoundingBox):
        self.root.insert(element)

    def remove(self, element: BoundingBox) -> bool:
        pass

    # more complex queries
    def nearest_neighbor(self, element: BoundingBox, k=1) -> \
            List[BoundingBox]:
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
