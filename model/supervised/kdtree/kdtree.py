import numpy as np
from utils.data_operation import euclidean_distance


class TreeNode:
    """
    Tree node class to build kdtree leaf
    """

    def __init__(self):
        self.left = None  # left node
        self.right = None  # right node
        self.feature_index = None  # feature index used to split tree
        self.split_value = None   # feature value used to split tree

    def __str__(self):
        return f"feature index: {self.feature_index}, split value: {self.split_value}."


class KDTree:
    def __init__(self, X):
        """
        KD Tree class

        Attributes:
            X: array type dataset (n_samples, n_features)
        """
        # calculate variance index list first
        self.variance_index_list = self.get_variance(X)
        # build KDtree recursively, start_split_index from 0 which is feature index with max variance
        self.root = self.build(X, 0)

    def get_variance(self, X):
        """
        Calculate variance for each feature and sorted variance get sorted index
        Args:
            X: array type dataset (n_samples, n_features)

        Returns:
            variance_index_list: list of feature index sorted base on variance desc (n_features)
        """
        variance_list = np.var(X, axis=0)
        variance_index_list = np.argsort(variance_list)
        return variance_index_list

    def build(self, X, start_split_index):
        """
        Recursively build KDtree. Here we use feature variance from max to min as split axis.
        Args:
            X: array type dataset (n_samples, n_features)
            start_split_index: int, index of variance_index_list, loop added over each recursive

        Returns:
            TreeNode: KDtree node contains split feature index, feature value, and left, right child
        """
        # when dataset is None which means we reach the leaf, then return None
        if X.shape[0] == 0:
            return None
        # get split feature index according to sorted feature variance
        split_feature_index = self.variance_index_list[start_split_index]
        # sort dataset asd base on split feature
        X.sort(key=lambda x: x[split_feature_index])
        # get median position index no matter X length is odd or even
        median_index = len(X) // 2
        # get dataset median value
        median = X[median_index]
        # create the tree node
        tree_node = TreeNode()
        # assign split feature index and split feature value
        tree_node.feature_index = split_feature_index
        tree_node.split_value = median
        # assign left and right child node
        # start_split_index + 1 which means use next biggest variance feature to split dataset
        tree_node.left = self.build(X[:median_index, :], start_split_index+1)
        tree_node.right = self.build(X[median_index:, :], start_split_index+1)

        return tree_node

    def _nearest_neighbor_search(self, tree_node, point):
        """
        Find nearest neighbor node of searching point, right now only support one point search each time.
        Inside class protected method, postorder traversal to search node
        Args:
            tree_node: KDtree node from build method
            point: array type (1, n_features)

        Returns:
            nearest_point: array type (1, n_features), node value in KDtree.
            nearest_dist: float, the distance between searching point and nearest_point.
        """
        # define recursive end case
        # when reach to None which means reach to leaf then return inf value as the nearest distance
        if tree_node is None:
            nearest_point = [0] * len(point)
            nearest_dist = float('inf')
            return nearest_point, nearest_dist

        # get this node split feature index and median value which used as split value when build tree
        split_feature_index = tree_node.feature_index
        median_point = tree_node.split_value

        # recursively go through KDtree, check go left or right
        # since KDtree is BST, so we can go left or right base on split point value
        if point[split_feature_index] <= median_point[split_feature_index]:
            nearest_point, nearest_dist = self._nearest_neighbor_search(tree_node.left, point)
        else:
            nearest_point, nearest_dist = self._nearest_neighbor_search(tree_node.right, point)

        # once get the leaf which is the nearest point of searching point, get new distance and nearest point
        cur_dist = euclidean_distance(point, median_point)
        # update new distance and nearest point if searching point close to split point than leaf point
        if cur_dist < nearest_dist:
            nearest_dist = cur_dist
            nearest_point = median_point.copy()
        # check if distance between split hyper plan and searching point is short then left point to searching point
        dist_to_plan = abs(point[split_feature_index] - median_point[split_feature_index])
        # check distance to hyper plan and compare the nearest dist and hyper plan distance
        # if distance to plan is farther than the nearest dist, which means the nearest point is current point
        # no need to search other side tree branch, otherwise other side tree branch contains the nearest point
        if dist_to_plan >= nearest_dist:
            return nearest_point, nearest_dist
        else:
            # if smaller than split point then which means left side already check then check brother side go right
            if point[split_feature_index] <= median_point[split_feature_index]:
                next_tree_node = tree_node.right
            else:
                next_tree_node = tree_node.left
            # recursively go through other side tree node
            new_nearest_point, new_nearest_dist = self._nearest_neighbor_search(next_tree_node, point)
            # check if other side new nearest distance is small than this side nearest distance
            # if it's smaller, then update the nearest point and nearest distance and return
            if new_nearest_point < nearest_dist:
                nearest_dist = new_nearest_dist
                nearest_point = new_nearest_point.copy()
            return nearest_point, nearest_dist

    def query(self, point):
        """
        Outside method to query nearest neighbor point and distance
        Args:
            point: array type (1, n_features)

        Returns:
            nearest_point: array type (1, n_features), node value in KDtree.
            nearest_dist: float, the distance between searching point and nearest_point.
        """

        nearest_point, nearest_dist = self._nearest_neighbor_search(self.root, point)
        return nearest_point, nearest_dist
