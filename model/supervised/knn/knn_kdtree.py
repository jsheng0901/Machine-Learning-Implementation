import numpy as np
import heapq


class Node:
    """ kd tree node class """

    def __init__(self, data, label, axis, depth=0, left=None, right=None):
        self.data = data  # split point original data on sample in whole data set, ex: [1, 2, 3, 9]
        self.depth = depth  # kd tree depth
        self.left = left  # left child node
        self.right = right  # right child node
        self.label = label  # split point corresponding label
        self.axis = axis  # split axis index


class KdTree:
    """
    kd tree algorithm
    first: select max variance dimension (feature) as split axis
    second: select median point in that dimension as split point
    keep repeated until no points in that range
    """

    def __init__(self, data, label, k):
        self.KdTree = None
        self.n = 0
        self.nearest = []
        self.create(data, label)
        self.k = k
        self.radius = float('inf')  # keep store min radius of target value and node

    def create(self, data, label, depth=0):
        """ build kd-tree """
        if len(data) <= 0:  # check if it's no points in that range, if no point we will stop at leave
            return None

        m, n = np.shape(data)
        self.n = n
        # axis = depth % self.n
        axis = np.argmax(np.var(data, axis=0))  # get largest variance feature (dimension) as split dimension
        mid = int(m // 2)  # get median as split point index
        data = sorted(data, key=lambda x: x[axis])  # sort the whole data used split dimension
        node = Node(data[mid], label[mid], axis, depth)
        if depth == 0:
            self.KdTree = node  # set root node
        node.left = self.create(data[:mid], label[:mid], depth + 1)  # left child, value < selected axis
        node.right = self.create(data[mid + 1:], label[mid + 1:], depth + 1)  # right child, value > axis

        return node

    def get_distance(self, node, x):
        """ get Euclidean distance """
        return np.sqrt(np.sum(np.square(x - node.data)))

    def traversal(self, node, x):
        """ post_order go through all node in kd_tree, and store distance between node and predict value """
        if node is None:
            return

        # check go to which branch
        if x[node.axis] < node.data[node.axis]:
            left = self.traversal(node.left, x)
            right = None
        else:
            left = None
            right = self.traversal(node.right, x)

        # get distance between node and x
        dist = self.get_distance(node, x)
        # update min heap to store closest node into heapq
        if len(self.nearest) < self.k or self.nearest[-1][0] > dist:
            heapq.heappush(self.nearest, (dist, node.label))
        # if len(self.nearest) > self.k:  # if more than k, then pop
        #     heapq.heappop(self.nearest)
        # update min distance of radius
        self.radius = min(self.radius, dist)
        # check if interaction between node split axis and target data axis
        if left is not None and right is None:
            if np.abs(node.data[node.axis] - x[node.axis]) < self.radius:
                right = self.traversal(node.right, x)
        elif left is None and right is not None:
            if np.abs(node.data[node.axis] - x[node.axis]) < self.radius:
                left = self.traversal(node.left, x)

        return dist

    def predict(self, X):
        """ search top nearest points """

        result = []
        for i in X:
            self.nearest = []
            self.traversal(self.KdTree, i)
            # get neighbors top small k
            labels = [n[1] for n in heapq.nsmallest(self.k, self.nearest, key=lambda x: x[0])]
            # count most frequent labels
            counts = np.bincount(labels.astype('int'))
            result.append(counts.argmax())

        return result
