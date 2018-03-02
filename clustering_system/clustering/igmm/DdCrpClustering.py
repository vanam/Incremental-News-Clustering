import math
from typing import List, Tuple, Callable

import numpy as np
import scipy.misc

from clustering_system.clustering.ClusteringABC import CovarianceType
from clustering_system.clustering.GibbsClusteringABC import GibbsClusteringABC
from clustering_system.clustering.gmm.GaussianMixtureABC import NormalInverseWishartPrior
from clustering_system.utils import draw_indexed
from clustering_system.visualization.LikelihoodVisualizer import LikelihoodVisualizer


def exponential_decay(d: float, a: float = 1):
    """
    Decays the probability of linking to an earlier customer exponentially
    with the distance to the current customer.

    f(d) = exp(-d / a) / a

    :param d: distance (non-negative finite value)
    :param a: decay constant
    :return: decay

    """
    return math.exp(-d / a)


def window_decay(d: float, a: float):
    """
    Only considers customers that are at most distance 'a' from the current customer.

    f(d) = 1/[d < a]

    :param d: distance (non-negative finite value)
    :param a: maximum distance
    :return: decay
    """
    return 1 if d < a else 0


def logistic_decay(d: float, a: float):
    """
    Logistic decay is a smooth version of the window decay.

    f(d) = 1 - 1 / (1 + exp(-d + a)) = exp(-d + a) / (1 + exp(-d + a))

    :param d: distance (non-negative finite value)
    :param a: the x-value of the sigmoid's midpoint
    :return: decay
    """
    return math.exp(-d + a) / (1 + math.exp(-d + a))


class DdCrpClustering(GibbsClusteringABC):

    def __init__(self, D: int, alpha: float, prior: NormalInverseWishartPrior, n_iterations: int,
                 decay_function: Callable[[float], float],
                 probability_threshold: float = 0.001,
                 visualizer: LikelihoodVisualizer = None,
                 covariance_type: CovarianceType = CovarianceType.full):
        super().__init__(D, alpha, prior, n_iterations, visualizer=visualizer, covariance_type=covariance_type)
        self.f = decay_function
        self.threshold = probability_threshold

        self.g = []  # undirected graph
        self.c = []  # customer assignments
        self.timestamps = []

        self.likelihood_cache = {}

    def add_documents(self, vectors: np.ndarray, metadata: np.ndarray):
        for md, vector in zip(metadata, vectors):
            doc_id, timestamp, *_ = md

            # Add document at the end of arrays
            self.ids.append(doc_id)
            self.c.append(self.N)                                       # Customer is assigned to self
            self.g.append({self.N})                                     # Customer has a link to self
            self.mixture.add(vector, self._get_new_cluster_number())    # Customer sits to his own table
            self.N += 1                                                 # Increment number of documents (customers)
            self.K += 1                                                 # Increment number of tables

            # Store timestamp of document (prior information)
            self.timestamps.append(timestamp / (60*60*24))  # Timestamp in days

    @property
    def likelihood(self) -> float:
        return self.mixture.likelihood

    @property
    def parameters(self) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        return self.mixture.parameters

    def _sample_document(self, i: int):
        # Remove customer assignment for a document i
        self._remove_assignment(i)

        # Calculate customer assignment probabilities for each document (including self)
        probabilities = self._get_assignment_probabilities(i)

        # Convert indexed log probabilities to probabilities (softmax)
        # print("----")
        # print(self.mixture.z)
        # print(probabilities)
        # print(scipy.misc.logsumexp(probabilities))
        # probabilities = np.exp(probabilities - scipy.misc.logsumexp(probabilities))
        # print(probabilities)

        ids, probabilities = zip(*probabilities)
        # print(probabilities)
        probabilities = np.exp(probabilities - scipy.misc.logsumexp(probabilities))
        # print(probabilities)
        probabilities = list(zip(ids, probabilities))

        # Sample new customer assignment
        # c = draw(probabilities)
        c = draw_indexed(probabilities)

        # Link document to new customer
        self._add_assignment(i, c)

    def __iter__(self):
        """
        For each document return (doc_id, cluster_id, linked_doc_id)
        """
        for doc_id, cluster_id, c_i in zip(self.ids, self.mixture.z, self.c):
            yield doc_id, cluster_id, self.ids[c_i]

    def _add_assignment(self, i: int, c: int):
        """
        Add new customer assignment c_i.

        :param i: document index
        :param c: new customer assignment (document index to which document i points to)
        """
        # If we have to join tables
        if self.mixture.z[i] != self.mixture.z[c]:
            # Move customers to a table with smaller cluster number
            if self.mixture.z[i] > self.mixture.z[c]:
                c_to_join = i
                table_to_join = self.mixture.z[i]
                table_no = self.mixture.z[c]
            else:
                c_to_join = c
                table_to_join = self.mixture.z[c]
                table_no = self.mixture.z[i]

            self.reusable_numbers.put_nowait(table_to_join)  # Make cluster number available
            self.K -= 1

            # Go through people at table with higher number and move them to the other table
            for c_c in self._get_people_next_to(c_to_join):
                self.mixture.z[c_c] = table_no

        # Set new customer assignment
        self.c[i] = c

        # Update undirected graph
        self.g[i].add(c)
        self.g[c].add(i)

    def _remove_assignment(self, i: int) -> None:
        """
        Remove customer assignment c_i.

        :param i: document index
        """
        is_split = self._is_table_split(i)

        # Update undirected graph
        c = self.c[i]
        c_c = self.c[c]

        if c == i:  # If self assignment
            self.g[i].remove(i)
        elif c_c == i:  # If trivial cycle a <--> b breaks to a <-- b
            pass  # Graph remains the same
        else:
            self.g[i].remove(c)
            self.g[c].remove(i)

        # Remove customer assignment c_i
        self.c[i] = -1

        if is_split:
            new_table_no = self._get_new_cluster_number()

            # Move customers to a new table
            for c in self._get_people_next_to(i):
                self.mixture.z[c] = new_table_no

            # Increment number of tables
            self.K += 1

    def _get_assignment_probabilities(self, i: int) -> List[Tuple[int, float]]:
        """
        Get probabilities of assignment of document i to all documents available.
        Probabilities lower than the threshold are not returned.
        Always at least one probability (self assignment) is returned.

        :param i: document index
        :return: list of tuples (document index, log assignment probability)
        """
        probabilities = []

        for c in range(self.N):
            prob = self._assignment_probability(i, c)

            # if prob < self.threshold:
            #     continue

            probabilities.append((c, prob))
            # probabilities.append(prob)

        return probabilities

    def _assignment_probability(self, i: int, c: int) -> float:
        """
        Return log probability of an assignment of document i to document c

        :param i: document index
        :param c: document index
        :return: Log probability of an assignment
        """
        # If self assignment
        if i == c:
            return math.log(self.alpha)

        # Time distance between documents
        d = abs(self.timestamps[i] - self.timestamps[c])

        # If not joining two tables
        if self.mixture.z[i] == self.mixture.z[c]:
            # Return ddCRP prior
            return math.log(self.f(d))
        else:
            return math.log(self.f(d)) + self._likelihood_under_z(i, c)

    def _is_table_split(self, i: int):
        """
        Does removal of c_i splits one table to two?

        :param i:
        :return: Return False if c_i is on a cycle, True otherwise
        """
        c = self.c[i]

        # If there is a trivial cycle a <--> a
        if c == i:
            return False

        # If there is a trivial cycle a <--> b
        if i == self.c[c]:
            return False

        # Traverse directed graph in search for a cycle which contains assignment c_i
        visited = {i}
        c = self.c[i]
        while c not in visited:
            visited.add(c)
            c = self.c[c]  # Every vertex has only one neighbour

            # Return true if next customer is a starting customer
            if c == i:
                return False

        return True

    def _get_people_next_to(self, c: int):
        """
        Get indices of customers siting with customer c at the same table

        :param c:
        :return:
        """
        # Traverse undirected graph from customer i
        visited = set()
        stack = [c]
        while len(stack) > 0:
            c = stack.pop()
            if c not in visited:
                visited.add(c)
                stack.extend(self.g[c] - visited)

        return visited

    def _likelihood_under_z(self, i: int, c: int):
        table_k_members = frozenset(self._get_people_next_to(i))
        table_l_members = frozenset(self._get_people_next_to(c))
        table_kl_members = frozenset(table_k_members.union(table_l_members))

        if table_k_members in self.likelihood_cache:
            table_k = self.likelihood_cache[table_k_members]
        else:
            table_k = self.mixture.get_marginal_likelihood(table_k_members)
            self.likelihood_cache[table_k_members] = table_k

        if table_l_members in self.likelihood_cache:
            table_l = self.likelihood_cache[table_l_members]
        else:
            table_l = self.mixture.get_marginal_likelihood(table_l_members)
            self.likelihood_cache[table_k_members] = table_l

        if table_kl_members in self.likelihood_cache:
            table_kl = self.likelihood_cache[table_kl_members]
        else:
            table_kl = self.mixture.get_marginal_likelihood(table_kl_members)
            self.likelihood_cache[table_kl_members] = table_kl

        # table_k = self._compute_marginal_likelihood(table_k_members)
        # table_l = self._compute_marginal_likelihood(table_l_members)
        # table_kl = self._compute_marginal_likelihood(table_kl_members)

        return table_kl - table_k - table_l
