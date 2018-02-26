import random
from typing import List, Tuple

import numpy as np
import scipy.misc

from clustering_system.clustering.ClusteringABC import CovarianceType
from clustering_system.clustering.GibbsClusteringABC import GibbsClusteringABC
from clustering_system.clustering.gmm.FullGaussianMixture import FullGaussianMixture
from clustering_system.clustering.gmm.GaussianMixtureABC import PriorABC
from clustering_system.utils import draw_indexed
from clustering_system.visualization.LikelihoodVisualizer import LikelihoodVisualizer


class DdCrpClustering(GibbsClusteringABC):

    def __init__(self, D: int, alpha: float, prior: PriorABC, n_iterations: int,
                 probability_threshold: float = 0.001,
                 K_max: int = None, visualizer: LikelihoodVisualizer = None,
                 covariance_type: CovarianceType = CovarianceType.full):
        super().__init__(D, alpha, prior, n_iterations, K_max=K_max, visualizer=visualizer)
        self.g = []  # undirected graph
        self.c = []  # customer assignments
        self.timestamps = []

        if covariance_type == CovarianceType.full:
            self.mixture = FullGaussianMixture(prior)
        else:
            raise NotImplementedError("Unsupported covariance type %s." % covariance_type)

    def add_documents(self, vectors: np.ndarray, metadata: np.ndarray):
        for md, vector in zip(metadata, vectors):
            doc_id, timestamp, *_ = md

            # Add document at the end of arrays
            self.ids.append(doc_id)
            self.X = np.vstack((self.X, np.array([vector])))
            self.c.append(self.N)    # customer is assigned to self
            self.z.append(self.K)    # customer sits to his own table
            self.g.append({self.N})  # customer has a link to self
            self.N += 1              # Increment number of documents (customers)
            self.K += 1              # Increment number of tables

            # Store timestamp of document (prior information)
            self.timestamps.append(timestamp)

    @property
    def likelihood(self) -> float:
        """
        :return: Return average log likelihood of data.
        """
        return random.random() * 50

    def _sample_document(self, i: int):
        # Remove customer assignment for a document i
        self._remove_assignment(i)

        # Calculate customer assignment probabilities for each document (including self)
        probabilities = self._get_assignment_probabilities(i)

        # Convert indexed log probabilities to probabilities (softmax)
        ids, probabilities = zip(*probabilities)
        probabilities = np.exp(probabilities - scipy.misc.logsumexp(probabilities))
        probabilities = list(zip(ids, probabilities))

        # Sample new customer assignment
        c = draw_indexed(probabilities)

        # Link document to new customer
        self._add_assignment(i, c)

    def __iter__(self):
        """
        For each document return (doc_id, cluster_id, linked_doc_id)
        """
        for doc_id, cluster_id, c_i in zip(self.ids, self.z, self.c):
            yield doc_id, cluster_id, self.ids[c_i]

    def _add_assignment(self, i: int, c: int):
        """
        Add new customer assignment c_i.

        :param i: document index
        :param c: new customer assignment (document index to which document i points to)
        """
        # If we have to join tables
        if self.z[i] != self.z[c]:
            # Move customers to a table z[i]
            table_no = self.z[i]

            for c_c in self._get_people_next_to(c):
                self.z[c_c] = table_no

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
        # Check if table assignment changes
        if self.c[i] == i:
            # Nothing to do if self assignment
            pass
        elif self._split_table(i):
            # Move customers to a new table K
            for c in self._get_people_next_to(i):
                self.z[c] = self.K

            # Increment number of tables
            self.K += 1

        # Remove customer assignment c_i
        self.c[i] = -1

    def _get_assignment_probabilities(self, i: int) -> List[Tuple[int, float]]:
        """
        Get assignment probabilities. Probabilities under the threshold are not returned.
        Always at least one probability (self assignment) is returned.

        :param i: document index
        :return: list of tuples (document index, assignment probability)
        """
        return [(i, 1.0), (i, 2.0)]

    def _split_table(self, i: int):
        """
        Does removal of c_i splits one table to two?

        :param i:
        :return: Return False if c_i is on a cycle, True otherwise
        """
        # If there is a trivial cycle a <--> b
        if self.c[i] == self.c[self.c[i]]:
            return False

        # Traverse directed graph in search for cycle which contains assignment c_i
        visited = {i}
        c = self.c[i]
        while c not in visited:
            visited.add(c)
            c = self.c[c]

            # Return true if next customer is a starting customer
            if c == i:
                return False

        return True

    def _get_people_next_to(self, c):
        """
        Get indices of customers siting with customer c at the same table

        :param c:
        :return:
        """
        # Traverse undirected graph from customer i
        visited = set()
        stack = [c]
        while stack:
            c = stack.pop()
            if c not in visited:
                visited.add(c)
                stack.extend(self.g[c] - visited)

        return visited
