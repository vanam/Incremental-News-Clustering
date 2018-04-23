from collections import defaultdict

from networkx import DiGraph, write_gexf


class GraphVisualizer:
    """Visualize clustering dynamics as a graph."""

    def __init__(self):
        self.documents = {}

    def add_documents(self, docs: list, metadata: list, time: int):
        """
        Add documents to a graph.

        :param docs: The document list
        :param metadata: The document metadata
        :param time: The time of visualization
        """
        for doc, metadata in zip(docs, metadata):
            doc_id = metadata[0]
            title = metadata[2]
            # For each document store (start_time, end_time, doc_vec, cluster_list, title, linked_doc_id_list)
            self.documents[str(doc_id)] = [time, time, doc, [], title, defaultdict(list)]

    def set_cluster_for_doc(self, t, doc_id, cluster_id, linked_doc_id: int = None):
        """
        Add document visualization attributes at time t.

        :param t: The time of visualization
        :param doc_id: The document id
        :param cluster_id: The cluster id
        :param linked_doc_id: The link to another document
        :return:
        """
        self.documents[doc_id][1] = t + 1
        self.documents[doc_id][3].append((int(cluster_id), t, t + 1))

        if linked_doc_id is not None:
            self.documents[doc_id][5][linked_doc_id].append((t, t + 1))

    def save(self, filename: str):
        """
        Save graph to GEXF file.

        :param filename: The filename
        """
        graph = DiGraph(mode="dynamic")

        for doc_id, doc in self.documents.items():
            graph.add_node(
                doc_id,
                label=doc_id[:7],
                start=doc[0],
                end=doc[1],
                cluster=doc[3],  # [(value, start, end), ...]
                title=doc[4],
                viz={"position": {"x": doc[2][0], "y": doc[2][1], "z": 0}}
            )

            # Add edges (customer assignments in dd-CRP)
            for linked_doc_id, spells in doc[5].items():
                graph.add_edge(
                    doc_id,
                    linked_doc_id,
                    spells=spells  # [(start, end), ...]
                )

        write_gexf(graph, filename)
