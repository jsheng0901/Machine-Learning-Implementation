from abc import abstractmethod
from typing import Dict, Union, Any, List
import numpy as np
from collections import defaultdict


class LSH(object):
    """
    LSH implementation

    Attributes:
        hash_size:
            The length of after apply hashing function
        num_hash_table:
            Number of hash tables.
            More table will lead to higher probability consider as same bucket
        bucket_length:
            User defined bucket length
            The bucket length can be used to control the average size of hash buckets
            and then the number of hash buckets. The larger bucket length increase the probability
            of feature being hashed to the same bucket which increase the TP, FP rate.
    """

    def __init__(self, hash_size: int, num_hash_table: int, bucket_length: int):
        self.hash_size = hash_size
        self.num_hash_table = num_hash_table
        self.bucket_length = bucket_length

    @abstractmethod
    def hash(self, **kwargs) -> Any:
        """
        This is hashing function which should be implemented in subclass
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, input_table: Union[List[List], np.array]) -> None:
        """
        Call fit function run LSH model

        Returns: None

        """
        raise NotImplementedError


class RandomProjectionLSH(LSH):
    """
    LSH implementation through random project bucket method
    """
    def __init__(self, hash_size, num_hash_table, bucket_length):
        super().__init__(hash_size, num_hash_table, bucket_length)
        self.buckets = [defaultdict(list) for _ in num_hash_table]

    def _init_uniform_planes(self, input_dim: int) -> np.array:
        """
        Get initial random uniform planes used for random project
        Args:
            input_dim: input table dimensions this will be reduced after project
        Returns:
            numpy array [input_dim, hash_size] for hash function
        """
        return np.random.randn(input_dim, self.hash_size)

    def _generate_uniform_planes_matrix(self, input_dim: int) -> np.array:
        """
        Generate array base uniform planes according to num of hash table
        Args:
            input_dim: input table dimensions this will be reduced after project
        Returns:
            numpy array [num_hash_table, input_dim, hash_size] for hash function
        """
        uniform_planes_matrix = []
        for _ in range(self.num_hash_table):
            uniform_planes_matrix.append(self._init_uniform_planes(input_dim))

        return np.array(uniform_planes_matrix)

    def hash(self, input_table: Union[List[List], np.array], projection_matrix: np.array) -> np.array:
        """
        This is random project bucket hashing function
        here we didn't convert to 0 1 case which convert + to 1, - to 0 after project.
        We use spark BucketRandomProjection hash way
        H = |[input table *(dot) project matrix] / bucket_length|
        Args:
            input_table: input table array
            projection_matrix: random uniform planes output
        Returns:
            numpy array [input_rows, hash_size] after hash function
        """
        if input_table.ndim == 2:
            input_table = np.expand_dims(input_table, axis=0)
        hash_results = np.floor(np.abs(np.matmul(input_table, projection_matrix) / self.bucket_length))
        return hash_results

    def insert(self, hash_results: np.array) -> None:
        """
        Insert each sample into same bucket.
        Here buckets will be list of dict, key is bucket name, value is list of sample index
        Args:
            hash_results: np array from hash function
        """
        for i in range(self.num_hash_table):
            for idx, val in enumerate(hash_results):
                self.buckets[i][np.array2string(val)].append(idx)

    def fit(self, input_table: Union[List[List], np.array]):
        """
        Hash input table and group into buckets
        Args:
            input_table: list of list or numpy array
        """
        # get input dimension and convert to numpy array
        _input_table_array = np.array(input_table)
        _, dims = np.shape(_input_table_array)

        # get init uniform planes matrix
        random_projection_matrix = self._generate_uniform_planes_matrix(dims)

        # calculate projection result after hashing
        hash_results = self.hash(_input_table_array, random_projection_matrix)

        # put hash results into same buckets for each table
        self.insert(hash_results)

    def query(self) -> Any:
        # TODO: implement query search later which could search most similar sample in buckets
        pass



