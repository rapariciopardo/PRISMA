import numpy as np
import random

__author__ = "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__license__ = "GPL"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"

# This work was partially based on openai/baselines available on https://github.com/openai/baselines/tree/tf2/baselines/deepq


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.total_samples = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        self.total_samples += 1
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(obs_t)
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(obs_tp1)
            dones.append(done)
        try:
            return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1, dtype=object), np.array(dones), np.ones(len(idxes), dtype=np.float32)
        except:
            print("ERROR")
            print(obses_t)
            raise(1)
    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
    
    def save(self, path):
        """Stores the replay buffer into a file

        Args:
            path (str): destination path
        """
        import pickle
        with open(path, "wb") as fp:
            pickle.dump(self._storage, fp)

import random
from typing import Optional, List, Union

import numpy as np

def unique(sorted_array):
    """
    More efficient implementation of np.unique for sorted arrays
    :param sorted_array: (np.ndarray)
    :return:(np.ndarray) sorted_array without duplicate elements
    """
    if len(sorted_array) == 1:
        return sorted_array
    left = sorted_array[:-1]
    right = sorted_array[1:]
    uniques = np.append(right != left, True)
    return sorted_array[uniques]


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """
        Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array that supports Index arrays, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.

        :param capacity: (int) Total size of the array - must be a power of two.
        :param operation: (lambda (Any, Any): Any) operation for combining elements (eg. sum, max) must form a
            mathematical group together with the set of possible values for array elements (i.e. be associative)
        :param neutral_element: (Any) neutral element for the operation above. eg. float('-inf') for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation
        self.neutral_element = neutral_element

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """
        Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        :param start: (int) beginning of the subsequence
        :param end: (int) end of the subsequences
        :return: (Any) result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # indexes of the leaf
        idxs = idx + self._capacity
        self._value[idxs] = val
        if isinstance(idxs, int):
            idxs = np.array([idxs])
        # go up one level in the tree and remove duplicate indexes
        idxs = unique(idxs // 2)
        while len(idxs) > 1 or idxs[0] > 0:
            # as long as there are non-zero indexes, update the corresponding values
            self._value[idxs] = self._operation(
                self._value[2 * idxs],
                self._value[2 * idxs + 1]
            )
            # go up one level in the tree and remove duplicate indexes
            idxs = unique(idxs // 2)

    def __getitem__(self, idx):
        assert np.max(idx) < self._capacity
        assert 0 <= np.min(idx)
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=np.add,
            neutral_element=0.0
        )
        self._value = np.array(self._value)

    def sum(self, start=0, end=None):
        """
        Returns arr[start] + ... + arr[end]

        :param start: (int) start position of the reduction (must be >= 0)
        :param end: (int) end position of the reduction (must be < len(arr), can be None for len(arr) - 1)
        :return: (Any) reduction of SumSegmentTree
        """
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """
        Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum for each entry in prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        :param prefixsum: (np.ndarray) float upper bounds on the sum of array prefix
        :return: (np.ndarray) highest indexes satisfying the prefixsum constraint
        """
        if isinstance(prefixsum, float):
            prefixsum = np.array([prefixsum])
        assert 0 <= np.min(prefixsum)
        assert np.max(prefixsum) <= self.sum() + 1e-5
        assert isinstance(prefixsum[0], float)

        idx = np.ones(len(prefixsum), dtype=int)
        cont = np.ones(len(prefixsum), dtype=bool)

        while np.any(cont):  # while not all nodes are leafs
            idx[cont] = 2 * idx[cont]
            prefixsum_new = np.where(self._value[idx] <= prefixsum, prefixsum - self._value[idx], prefixsum)
            # prepare update of prefixsum for all right children
            idx = np.where(np.logical_or(self._value[idx] > prefixsum, np.logical_not(cont)), idx, idx + 1)
            # Select child node for non-leaf nodes
            prefixsum = prefixsum_new
            # update prefixsum
            cont = idx < self._capacity
            # collect leafs
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=np.minimum,
            neutral_element=float('inf')
        )
        self._value = np.array(self._value)

    def min(self, start=0, end=None):
        """
        Returns min(arr[start], ...,  arr[end])

        :param start: (int) start position of the reduction (must be >= 0)
        :param end: (int) end position of the reduction (must be < len(arr), can be None for len(arr) - 1)
        :return: (Any) reduction of MinSegmentTree
        """
        return super(MinSegmentTree, self).reduce(start, end)


class ReplayBuffer_(object):
    def __init__(self, size: int):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.total_samples = 0

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def storage(self):
        """[(Union[np.ndarray, int], Union[np.ndarray, int], float, Union[np.ndarray, int], bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self) -> int:
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples: int) -> bool:
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self) -> int:
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done
        """
        self.total_samples += 1
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def extend(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new batch of transitions to the buffer

        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch

        Note: uses the same names as .add to keep compatibility with named argument passing
                but expects iterables and arrays with more than 1 dimensions
        """
        for data in zip(obs_t, action, reward, obs_tp1, done):
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    @staticmethod
    def _normalize_obs(obs: np.ndarray) -> np.ndarray:
        """
        Helper for normalizing the observation.
        """
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray) -> np.ndarray:
        """
        Helper for normalizing the reward.
        """
        return reward

    def _encode_sample(self, idxes: Union[List[int], np.ndarray],):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return (self._normalize_obs(np.array(obses_t)),
                np.array(actions),
                self._normalize_reward(np.array(rewards)),
                self._normalize_obs(np.array(obses_tp1)),
                np.array(dones))

    def sample(self, batch_size: int, **_kwargs):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer_):
    def __init__(self, size, alpha, nb_neighbors, index):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha
        
        ## store indexes for each neighbors
        self.nb_neighbors = nb_neighbors
        self.index = index
        self.neighbors_idx = [[] for _ in range(self.nb_neighbors)]
        self.latest_gradient_step = [1]*self.nb_neighbors
        self.original_prios = []
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_t, action, reward, obs_tp1, done, prio):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        :param prio: (float) is the weight of the observation
        """
        idx = self._next_idx
        super().add(obs_t, action, reward, obs_tp1, done)
        self.original_prios.append(prio)
        self.neighbors_idx[action].append(idx)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def extend(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new batch of transitions to the buffer

        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch

        Note: uses the same names as .add to keep compatibility with named argument passing
            but expects iterables and arrays with more than 1 dimensions
        """
        idx = self._next_idx
        super().extend(obs_t, action, reward, obs_tp1, done)
        while idx != self._next_idx:
            self._it_sum[idx] = self._max_priority ** self._alpha
            self._it_min[idx] = self._max_priority ** self._alpha
            idx = (idx + 1) % self._maxsize

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, len(self._storage) - 1)
        # TODO(szymon): should we ensure no repeats?
        mass = np.random.random(size=batch_size) * total
        # print(total, self._it_sum._value.shape)
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size: int, beta: float = 1):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        # idxes = self._sample_proportional(batch_size)
        # weights = []
        # p_min = self._it_min.min() / self._it_sum.sum()
        
        # max_weight = (p_min * len(self._storage)) ** (-beta)
        # p_sample = self._it_sum[idxes] / self._it_sum.sum()
        idxes = np.array([random.randint(0, len(self._storage) - 1) for _ in range(batch_size)])
        weights = (self._it_sum[idxes] / self._it_sum.sum()) * len(self._storage)
        
        # print("sample" , self.index, weights, self._it_sum[idxes])
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights])
    
    def save(self, path):
        """Stores the replay buffer into a file

        Args:
            path (str): destination path
        """
        import pickle
        with open(path, "wb") as fp:
            pickle.dump(self._storage, fp)
            
    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        # assert len(idxes) == len(priorities)
        # assert np.min(priorities) > 0
        idxes = np.array(idxes, dtype=int)
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.storage)
        values = (np.array(self.original_prios)[idxes]/ self.latest_gradient_step[priorities] )** self._alpha
        self._it_sum[idxes] = values
        self._it_min[idxes] = values
        self._max_priority = max(self._max_priority, np.max(values))