from networkx import DiGraph, write_gexf


class GraphVisualizer:

    def __init__(self):
        self.documents = {}

    def add_documents(self, docs: list, metadata: list, time: int):
        for doc, metadata in zip(docs, metadata):
            doc_id = metadata[0]
            title = metadata[2]
            # For each document store (start_time, end_time, doc_vec, cluster_list, title, linked_doc_id)
            self.documents[str(doc_id)] = [time, time, doc, [], title, None]

    def set_cluster_for_doc(self, t, doc_id, cluster_id, linked_doc_id: int = None):
        self.documents[doc_id][1] = t + 1
        self.documents[doc_id][3].append((int(cluster_id), t, t + 1))
        self.documents[doc_id][5] = linked_doc_id

    def save(self, filename: str):
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
            graph.add_edge(doc_id, doc[5])

        write_gexf(graph, filename)
