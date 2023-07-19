import numpy as np
from sklearn.neighbors import KDTree
from utils.data_operation import euclidean_distance


class UnionFind:
    """
    UnionFind to implement hierarchical single linkage tree
    This UnionFind is a little different with traditional UnionFind,
    because we have to keep record size of cluster and track the parent node for both node

    Parameters:
    -----------
    n: int
        The number of nodes not number of edges
    """
    def __init__(self, n):
        # initial parent arr with number of nodes and number of edges
        self.parent = -1 * np.ones(2 * n - 1, dtype=np.int)
        # initial size array, first n is initial as 1 since each leaf node is size 1,
        # next number of edge as 0, this is what we will loop through each edge in MST and record current size
        self.size = np.hstack((np.ones(n, dtype=np.int), np.zeros(n - 1, dtype=np.int)))
        # this is initial start index to record parent node, like node 0 and node 1 will connect to node next label
        self.next_label = n

    def find(self, x):
        """ find last parent node value and give each connected node same parent label"""
        # initial start point
        p = x
        # find last parent node until index in parent arr is -1 which is initial value
        while self.parent[x] != -1:
            x = self.parent[x]
        # label each node start from initial point p to last parent node
        # for leaf node this will point to -1 index in parent arr, so last index in parent will always record
        # last leaf node value
        while self.parent[p] != x:
            p, self.parent[p] = self.parent[p], x

        # return parent node value
        return x

    def union(self, x, y):
        """ record current node cluster size and give label to input node """
        # record current node cluster size, like dynamic programming, next label position in size arr
        # will record current two node x, y formed cluster size sum.
        self.size[self.next_label] = self.size[x] + self.size[y]
        # give label to x, y node, for leaf, x, y usually be itself like 2 --> 2, for any other, x, y always
        # be their parent node value, note, in here starting label is index of number of nodes,
        # like 10 (10 points in over all dataset), ex: x:3 y:4 n:10 then next_label: 10, node3 and node 4 point label 10
        self.parent[x] = self.next_label
        self.parent[y] = self.next_label
        # rolling to next index
        self.next_label += 1

        return


class HDBSCAN:
    """
    Basin implementation of HDBSCAN clustering method
    HDBSCAN good for any shape of clusters, including non-convex dataset,
    better than KMeans when dataset have lots of noise data.
    Better than DBSCAN to find clusters of varying densities,
    and be more robust to parameter selection.

    Parameters:
    -----------
    min_cluster_size : int, optional (default=5)
        The minimum size of clusters; single linkage splits that contain
        fewer points than this will be considered points "falling out" of a
        cluster rather than a cluster splitting into two new clusters.
        Usually large size will result as small number of clusters and more compact cluster shape

    min_samples : int, optional (default=5)
        The number of samples in a neighbourhood for a point to be
        considered a core point. Usually large value will cause data push away and more data
        consider as noise.
    """
    def __init__(self, min_cluster_size: int = 5, min_samples: int = 5):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def get_mutual_reachability_distance(self, X, core_distance) -> np.array:
        """
        Get mutual reachability distance, which define as mrd(a, b) = max{core_k_(a), core_k_(b), d(a,b)}
        Args:
            core_distance: array like (n_samples, )
            X : array type dataset (n_samples, n_features)

        Returns:
            array like matrix (n_sample, n_sample)
        """
        n_sample = X.shape[0]
        mrd_matrix = np.zeros((n_sample, n_sample))
        # loop through each point and compare
        for i in range(n_sample):
            for j in range(n_sample):
                # get two points distance
                distance = euclidean_distance(X[i, :], X[j, :])
                mrd_matrix[i, j] = max(core_distance[i], core_distance[j], distance)

        return mrd_matrix

    def generate_min_spanning_tree(self, mrd_matrix: np.array) -> np.array:
        """
        A minimum spanning tree (MST) or minimum weight spanning tree is a subset of the edges of a connected,
        edge-weighted undirected graph that connects all the vertices together,
        without any cycles and with the minimum possible total edge weight.
        That is, it is a spanning tree whose sum of edge weights is as small as possible.
        More generally, any edge-weighted undirected graph (not necessarily connected) has a minimum spanning forest,
        which is a union of the minimum spanning trees for its connected components.

        Generate minimum spanning tree, O(V^2) this step, V is number of edges
        Args:
            mrd_matrix: numpy array like matrix (n_sample, n_sample)

        Returns:
            mst, numpy array like list (n_sample - 1, 3)
            n_sample - 1 = number of edge
            3 means (node_index, node_index, distance)
        """
        mst = []
        n_sample = mrd_matrix.shape[0]
        visit = [False] * n_sample

        # start from random sample (node)
        curr_index = np.random.choice(n_sample)
        # mark first initial sample
        visit[curr_index] = True
        # initial next index pointer
        next_index = curr_index
        # loop until all point visited
        while not all(visit):
            # set each time loop initial min distance
            min_distance = float('inf')
            for i in range(n_sample):
                d = mrd_matrix[curr_index, i]
                # check if visited and distance
                if not visit[i] and d < min_distance:
                    next_index = i
                    min_distance = d
            # mark next index as visited
            visit[next_index] = True
            # add node linkage into mst
            mst.append([curr_index, next_index, min_distance])
            # set curr as next
            curr_index = next_index

        return np.array(mst)

    def generate_single_linkage_tree(self, min_spanning_tree):
        """
        Convert edge list min spanning tree into standard hierarchical clustering format
        Args:
            min_spanning_tree: array like list (n_sample - 1 -> n_edge, 3)

        Returns:
            slt_arr: array like list (n_sample - 1 -> n_edge, 4),
                     4 meas parent_start_node index, parent_end_node index, MRD, current cluster size include itself
        """
        # define final single linkage tree result
        (n_edge, n_dim) = min_spanning_tree.shape
        n_sample = n_edge + 1
        slt_arr = np.zeros((n_edge, n_dim + 1))
        # initial union find structure
        union_find = UnionFind(n_sample)
        # loop through MST edges
        for index in range(n_edge):
            # get each edges, start node, end node and mrd distance between nodes
            node1 = int(min_spanning_tree[index, 0])
            node2 = int(min_spanning_tree[index, 1])
            dist = min_spanning_tree[index, 2]
            # find each node parent nodes
            p_node1, p_node2 = union_find.find(node1), union_find.find(node2)
            # assign value to SLT result array
            slt_arr[index][0] = p_node1
            slt_arr[index][1] = p_node2
            slt_arr[index][2] = dist
            slt_arr[index][3] = union_find.size[p_node1] + union_find.size[p_node2]
            # union this two nodes parent's node
            union_find.union(p_node1, p_node2)

        return slt_arr

    def bfs_for_hierarchy(self, single_linkage_tree, root):
        """
        BFS for single linkage tree, loop through tree each layer by layer from top to bottom. Use queue.
        Args:
            single_linkage_tree: ndarray (n_samples - 1, 4)
            root: int, new index of root from single linkage tree create

        Returns: list, (2 * n_edges + 1)
            list of node index value in single linkage tree array, from root to leaf
        """
        # get number of sample first, prepare for later use
        n_samples = single_linkage_tree.shape[0] + 1
        # initial queue for bfs
        queue = [root]
        # initial result array
        result = []
        # loop through queue, until queue is empty
        while queue:
            # fast way to combine each level tree node together
            ##############################################################################
            # result.extend(queue)
            # queue = [q - n_samples for q in queue if q >= n_samples]
            # if queue:
            #     queue = single_linkage_tree[queue, :2].flatten().astype(np.int).tolist()
            ##############################################################################

            # another traditional bfs way
            first = queue.pop(0)
            # add into final result
            result.append(first)
            # check if this index is leaf node or not, if it's leaf, then stop, this is the most important step
            # this step inherent from how to build single linkage tree, if it's leaf then no need convert index.
            # (the index value of node inside SLT - number of points) is index of next connected node in SLT array
            if first >= n_samples:
                # convert index
                index_of_next_node = first - n_samples
                # get child node value by using convert index
                child_nodes_value = single_linkage_tree[index_of_next_node, :2].flatten().astype(np.int).tolist()
                # add into queue
                queue.extend(child_nodes_value)

        return result

    def condense_tree(self, single_linkage_tree):
        """
        Condense a tree according to a minimum cluster size. The result is a much simpler
        tree. Include extra information on the lambda value at which individual points depart clusters for later
        analysis and computation.

        Parameters
        ----------
        single_linkage_tree : ndarray (n_samples, 4)
            A single linkage hierarchy tree from generate_single_linkage_tree output.

        Returns
        -------
        condensed_tree : numpy array  (n_samples, 4)
            Effectively an edge list with a parent (new label array value from next label),
            child (if leaf then old value else new label value), lambda_val
            and child_size in each row providing a tree structure.
        """
        # initial root index which equal to number of all nodes (each edge contains two nodes, will duplicate)in SLT
        root = 2 * single_linkage_tree.shape[0]
        n_samples = single_linkage_tree.shape[0] + 1
        next_label = n_samples + 1
        # get all node index list from root to leaf
        node_list = self.bfs_for_hierarchy(single_linkage_tree, root)
        # create a new label list equal to all nodes inside SLT + one root
        new_label = np.empty(root + 1, dtype=np.int)
        # give the root index in new label list a new label
        new_label[root] = n_samples
        result_list = []
        # create a list to record if this node is visited or to keep split
        ignore = np.zeros(len(node_list), dtype=np.int)

        for node in node_list:
            # if node is visited before, or it's leaf then continue
            if ignore[node] or node < n_samples:
                continue
            # get node children node in SLT
            children = single_linkage_tree[node - n_samples]
            left = int(children[0])
            right = int(children[1])
            mrd = children[2]
            # calculate lambda value which will be used later
            lambda_value = 1 / mrd if mrd > 0 else float('inf')
            # count how many node after this edge split into two group, this is not children[3]
            # children[3] is created during SLT by union find, which means when union those two nodes how many total
            # nodes we will have, but in here we want to do inverse, if split this two union nodes, how many nodes
            # we will have on left and right. So we need find left, right node child node's total number nodes in SLT
            left_count = single_linkage_tree[left - n_samples][3] if left >= n_samples else 1
            right_count = single_linkage_tree[right - n_samples][3] if right >= n_samples else 1
            # case 1: after split, both child contains number of nodes > min cluster size --> keep split children node
            if left_count >= self.min_cluster_size and right_count >= self.min_cluster_size:
                # give next label for split node in new label list, same as in build SLT
                new_label[left] = next_label
                # add one for next label
                next_label += 1
                result_list.append((new_label[node], new_label[left], lambda_value, left_count))
                # same as left side
                new_label[right] = next_label
                next_label += 1
                result_list.append((new_label[node], new_label[right], lambda_value, right_count))
            # case 2: after split, both child contains number of nodes < min cluster size --> stop split children node
            elif left_count < self.min_cluster_size and right_count < self.min_cluster_size:
                # get left side all children nodes
                left_node_list = self.bfs_for_hierarchy(single_linkage_tree, left)
                for sub_node in left_node_list:
                    # if node is leaf during SLT created step (trick part inside SLT created)
                    if sub_node < n_samples:
                        # result list:
                        # 1. add this node new parent node value, all parent node value will be same since stop split
                        # 2. this node value,
                        # 3. and when split to create this branch, the edge lambda value,
                        # 4. left cluster size as 1
                        result_list.append((new_label[node], sub_node, lambda_value, 1))
                    # mark each sub node visited
                    ignore[sub_node] = True
                # do same thing on right side
                right_node_list = self.bfs_for_hierarchy(single_linkage_tree, right)
                for sub_node in right_node_list:
                    if sub_node < n_samples:
                        result_list.append((new_label[node], sub_node, lambda_value, 1))
                    ignore[sub_node] = True
            # case 3: after split, left child contains number of nodes < min cluster size --> keep split right node
            elif left_count < self.min_cluster_size:
                # give keep split node in new label with parent node value
                new_label[right] = new_label[node]
                left_node_list = self.bfs_for_hierarchy(single_linkage_tree, left)
                for sub_node in left_node_list:
                    if sub_node < n_samples:
                        result_list.append((new_label[node], sub_node, lambda_value, 1))
                    ignore[sub_node] = True
            # case 4: after split, right child contains number of nodes < min cluster size --> keep split left node
            else:
                # give keep split node in new label with parent node value
                new_label[left] = new_label[node]
                right_node_list = self.bfs_for_hierarchy(single_linkage_tree, right)
                for sub_node in right_node_list:
                    if sub_node < n_samples:
                        result_list.append((new_label[node], sub_node, lambda_value, 1))
                    ignore[sub_node] = True

        return np.array(result_list, dtype=[('parent', np.int), ('child', np.int),
                                            ('lambda_val', float), ('child_size', np.int)])

    def compute_stability(self, condensed_tree):
        """
        Computer stability for each cluster from condense tree.
        Formular： sum((lambda p - lambda birth) * child size, each node inside cluster)

        Args:
            condensed_tree: numpy array  (n_samples, 4)
                A condensed tree from condensed_tree function
        Returns:
            dictionary  key: cluster parent node value, value: stability of this cluster
        """
        # find the largest child node value
        largest_child = condensed_tree['child'].max()
        # find the smallest cluster node value, parent node value represent for each node parent in tree
        # each distinct parent value represent for one cluster
        smallest_cluster = condensed_tree['parent'].min()
        largest_cluster = condensed_tree['parent'].max()
        # max parent node value - min parent node value add itself equal to number of clusters in tree
        # because only when left, right after split both side number > min number clusters will +1 parent node value
        num_clusters = largest_cluster - smallest_cluster + 1
        # in this case, the smallest parent node value > max child value which means is not split in tree,
        # which means entire tree is one cluster
        if largest_child < smallest_cluster:
            # change child to min parent node value which is also root (max) value
            # in order to record root lambda birth value
            largest_child = smallest_cluster
        # sort child and lambda value according to child node value
        sorted_child_data = np.sort(condensed_tree[['child', 'lambda_val']], axis=0)
        births = np.nan * np.ones(largest_child + 1)
        sorted_children = sorted_child_data['child'].copy()
        sorted_lambdas = sorted_child_data['lambda_val'].copy()

        parents = condensed_tree['parent']
        sizes = condensed_tree['child_size']
        lambdas = condensed_tree['lambda_val']
        # initial child and lambda value, here like linked node dummy head, initial is node before the actual first node
        current_child = -1
        min_lambda = 0
        # from bottom to top loop all node assign lambda birth value
        for row in range(sorted_child_data.shape[0]):
            child = sorted_children[row]
            _lambda = sorted_lambdas[row]
            # we use linked node traver way, child and _lambda are always one step before curr node
            # ex: when child, _lambda in second node (1, 1), curr, min_lambda still in previous node (0, 1)
            if child == current_child:
                min_lambda = min(min_lambda, _lambda)
            # when begin loop not initial null node, we begin assign value to previous step (curr node)
            elif current_child != -1:
                # lambda birth equal to when split edge created this node cluster, same index lambda value in SLT
                births[current_child] = min_lambda
                # update curr and min value to real loop step which are child, _lambda
                current_child = child
                min_lambda = _lambda
            # we assign first node value to null head, which means pointer from null head to first node
            else:
                current_child = child
                min_lambda = _lambda
        # case the last node not loop in previous loop, we manually assign last node value to birth array
        if current_child != -1:
            births[current_child] = min_lambda
        births[smallest_cluster] = 0    # this is root value birth lambda, which is 0

        result = np.zeros(num_clusters)
        # from top to bottom node calculate all clusters stability
        for i in range(condensed_tree.shape[0]):
            parent = parents[i]
            lambda_ = lambdas[i]
            child_size = sizes[i]
            result_index = parent - smallest_cluster
            # calculate stability for each cluster sum all node stability, different  is times size but not in document
            # lambda_ equal lambda p in document, birth equal to lambda birth
            # but personally think lambda_ same as lambda death, when node split fall out of cluster
            result[result_index] += (lambda_ - births[parent]) * child_size
        # return as dict format, transfer to 2d array first
        result_pre_dict = np.vstack((np.arange(smallest_cluster, largest_cluster + 1), result)).T

        return dict(result_pre_dict)

    def get_clusters(self, condensed_tree, stability_dict):
        """
        Given a tree and stability dict, produce the cluster labels (probabilities)
        Args:
            condensed_tree: numpy array  (n_samples, 4)
                A condensed tree from condensed_tree function
            stability_dict: dictionary
                key: cluster id, value: S_lambda stability of this cluster
        Returns:
            labels: ndarray (n_samples)
                An integer array of cluster labels, with -1 label as noise
            probabilities: ndarray (n_samples)
                The cluster membership strength of each sample
            stabilities: ndarray (n_clusters)
                The cluster coherence strength of each cluster
        """
        node_list = sorted(stability_dict.keys(), reverse=True)
        cluster_tree = condensed_tree[condensed_tree['child_size'] > 1]
        is_cluster = {cluster: True for cluster in node_list}
        max_lambda = np.max(condensed_tree['lambda_val'])

        cluster_sizes = {child: child_size for child, child_size in zip(cluster_tree['child'], cluster_tree['child_size'])}
        # Compute cluster size for the root node
        cluster_sizes[node_list[-1]] = np.sum(cluster_tree[cluster_tree['parent'] == node_list[-1]]['child_size'])

        for node in node_list:
            child_selection = (cluster_tree['parent'] == node)
            subtree_stability = np.sum([
                stability_dict[child] for
                child in cluster_tree['child'][child_selection]])
            # case1: current node stability smaller than sum of child node stability,
            # then set current node stability as sum of child node stability
            if subtree_stability > stability_dict[node]:
                is_cluster[node] = False
                stability_dict[node] = subtree_stability
            # case2: current node stability larger than sum of child node stability,
            # then keep current node as one cluster and delete child nodes
            else:
                for sub_node in self.bfs_from_cluster_tree(cluster_tree, node):
                    if sub_node != node:
                        is_cluster[sub_node] = False

        clusters = set([c for c in is_cluster if is_cluster[c]])
        cluster_map = {c: n for n, c in enumerate(sorted(list(clusters)))}
        reverse_cluster_map = {n: c for c, n in cluster_map.items()}

        labels = self.do_labelling(condensed_tree, clusters, cluster_map)
        probabilities = self.get_probabilities(condensed_tree, reverse_cluster_map, labels)
        stabilities = self.get_stability_scores(labels, clusters, stability_dict, max_lambda)

        return labels, probabilities, stabilities

    def fit(self, X):
        """
        Perform HDBSCAN clustering from features or distance matrix.

        Parameters
        ----------
        X : array type dataset (n_samples, n_features)

        Returns
        -------
        self : object
            Returns self
        """
        # check min sample value
        size = X.shape[0]
        self.min_samples = min(self.min_samples, size - 1)
        # Apply KDTree to get knn tree graph
        # TODO implement KDTree later
        tree = KDTree(X)
        # Get distance to kth nearest neighbour core_k_(a) in paper, get [n_samples, ]
        # min_sample will include itself, so add 1 when find kth neighbor
        core_distances = tree.query(X, k=self.min_samples + 1)[0][:, -1].copy()
        # build mutual reachability distance matrix
        mrd_matrix = self.get_mutual_reachability_distance(X, core_distances)
        # build minimum spanning tree
        min_spanning_tree = self.generate_min_spanning_tree(mrd_matrix)
        # Sort edges of the min_spanning_tree by weight (mrd) from min -> max
        min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]
        # build hierarchical clustering
        single_linkage_tree = self.generate_single_linkage_tree(min_spanning_tree)
        # condense the hierarchical single linkage tree into a cluster tree
        condensed_tree = self.condense_tree(single_linkage_tree)
        # compute stability for each cluster
        stability_dict = self.compute_stability(condensed_tree)
        # get clusters result
        # TODO finish implement get clusters later
        labels, probabilities, stabilities = self.get_clusters(condensed_tree, stability_dict)

        return labels, probabilities, stabilities


# test
# from sklearn.datasets import make_blobs
#
# data, _ = make_blobs(5, 2)
#
# clusterer = HDBSCAN(min_cluster_size=3)
# cluster_labels = clusterer.fit(data)


