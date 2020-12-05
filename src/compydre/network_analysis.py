import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from scipy import stats


class NetworkAnalysis(object):
    def __init__(
        self, species, species_class, common_name, comadre_reader=None, asnr_reader=None
    ):
        self.species = species
        self.species_class = species_class
        self.common_name = common_name
        self.comadre_reader = comadre_reader
        self.asnr_reader = asnr_reader.species_data
        self._comadre_report = None
        self._asnr_report = None
        self._dual_report = None

    def create_comadre_report(self):
        print("Gathering three random matrix population models...")
        print("Projected population growth for {}".format(self.species))
        print("Lifecycle graphs for {}".format(self.species))
        return

    @staticmethod
    def print_asnr_metadata(meta_dict):
        pos = 0

        # get longest key
        maxi = 0
        for e in meta_dict:
            if len(e) > maxi:
                maxi = len(e)

        if maxi < pos:
            maxi = pos

        # print dictionary with alignment
        for e in meta_dict:
            if e not in ["Attributes Available", "Note", "Citation"]:
                print(e + " : " + " " * (maxi - len(e)) + meta_dict[e])

    def create_asnr_graph(self, graph):
        g = graph
        weights = list(nx.get_edge_attributes(g, "weight").values())
        centralities = np.array(list(nx.degree_centrality(g).values()))
        nodes = g.nodes()

        pos = nx.spring_layout(g)
        plt.figure(figsize=(15, 10))
        ec = nx.draw_networkx_edges(
            g, pos, edge_color=weights, alpha=0.2, edge_cmap=plt.cm.Reds, width=5
        )
        nc = nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=nodes,
            node_color=centralities,
            node_size=abs(stats.zscore(centralities) * 1000),
            cmap=plt.cm.Blues,
        )

        cbar = plt.colorbar(nc, fraction=0.025, pad=0.04)
        cbar.set_label("Node Degree Centrality", fontsize=18)
        cbar2 = plt.colorbar(ec, fraction=0.04, orientation="horizontal", pad=0.04)
        cbar2.set_label("Edge Weight", fontsize=18)
        plt.title("Social Contact Network: {}".format(self.species))
        plt.axis("off")
        plt.show()

    def create_asnr_report(self):
        print(
            "Creating ASNR report for {} with the common name {}".format(
                self.species, self.common_name
            )
        )
        print("Finding summary statistics for the {} graphs found...")
        data = self.asnr_reader[self.species_class][self.common_name]
        print(
            "There are {} NetworkX graphs available for {}".format(
                len(data["graphs"]), self.species
            )
        )
        meta_dict = data["metadata"]
        self.print_asnr_metadata(meta_dict)
        print("Showing graph for the social contact graph")
        self.create_asnr_graph(data["graphs"][0])
        return

    def show_reports(self):
        self.create_comadre_report()
        self.create_asnr_report()
