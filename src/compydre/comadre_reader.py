import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from rpy2.robjects import r, pandas2ri


MATRIX_TYPES = ["matA", "matU", "matF", "matC"]


class SpeciesMatrix(object):
    def __init__(self, species, matrix_dict, study_info, metadata, index):
        self.species = species
        self.mat_a = matrix_dict["matA"]
        self.mat_u = matrix_dict["matU"]
        self.mat_f = matrix_dict["matF"]
        self.mat_c = matrix_dict["matC"]
        self.metadata = metadata
        self.index = index
        self.study_info = study_info
        self._lc_graph = None

    def project_species_growth(self, n_years):
        n = np.array([100 for i in range(0, len(self.study_info))])
        population = {0: n}
        for i in range(1, n_years + 1):
            new_n = self.mat_a.dot(n)
            population[i] = new_n
            n = new_n
        return population

    def return_lc_graph(self):
        pop_types = self.study_info["StudyPopulationGroup"].values
        G = nx.MultiGraph()
        mat_shape = self.mat_a.shape
        for i in range(0, mat_shape[0]):
            for j in range(0, mat_shape[1]):
                per_cap_rate = self.mat_a[i][j]
                if per_cap_rate > 0:
                    G.add_edge(pop_types[j], pop_types[i], weight=per_cap_rate, labels=True)
        return G

    def show_lc_graph(self):
        plt.figure(figsize=(15, 10))
        nx.draw_networkx(self.lc_graph, node_size=1000, font_weight="bold")
        plt.axis("off")
        plt.show()

    @property
    def lc_graph(self):
        if self._lc_graph is None:
            self._lc_graph = self.return_lc_graph()
        return self._lc_graph


class ComadreReader(object):
    def __init__(self, path):

        self.path = path

        ## Matrix data by species
        self._raw_data = None
        self._metadata = None
        self._species_data = None

    def read_comadre(self):
        r.load(self.path)
        self._raw_data = r.comadre

    def return_species(self, species):
        try:
            species_data = self.species_data[species]
            return species_data

        except:
            similar = list(
                map(
                    self.species_data.get,
                    filter(lambda x: species in x, self.species_data),
                )
            )
            print("Unable to find the species you entered. Did you mean one of these?")
            print(similar)
            raise Exception("Species not in Comadre Database")

    def return_matrices(self, species_index, matrix_index):
        mat = self.raw_data.rx2("mat")[species_index][matrix_index]
        np_arr = np.array(mat)
        return np_arr

    def return_study_info(self, species_index):
        study_info = pd.DataFrame(
            np.array(self.raw_data.rx2("matrixClass")[species_index]).T,
            columns=[
                "MatrixClassOrganized",
                "MatrixClassAuthor",
                "MatrixClassNumber",
                "MatrixId",
            ],
        )
        study_info = study_info[["MatrixClassAuthor", "MatrixId"]]
        study_info.columns = ["StudyPopulationGroup", "MatrixId"]
        return study_info

    def return_metadata(self):
        metadata = pd.DataFrame(
            np.array(self.raw_data.rx2("metadata")).T,
            columns=list(r.names(self.raw_data.rx2("metadata"))),
        )
        return metadata

    def create_species_dict(self):
        species_dict = {}
        curr = 0
        print("Reading in data from specified path")
        species_list = self.raw_data.rx2("metadata").rx2("SpeciesAccepted")
        print("Gathering study metadata")
        self._metadata = self.return_metadata()

        for i in species_list:
            if i not in species_dict:

                species_dict[i] = []

            matrices = {
                MATRIX_TYPES[i]: self.return_matrices(curr, i) for i in range(0, 4)
            }
            index = curr
            study_info = self.return_study_info(curr)
            metadata = self._metadata.iloc[curr]

            species_dict[i].append(
                SpeciesMatrix(i, matrices, study_info, metadata, index)
            )
            curr += 1
        print("Compadre data is now accessible")
        return species_dict

    @property
    def species_data(self):
        if self._species_data is None:
            self._species_data = self.create_species_dict()
        return self._species_data

    @property
    def raw_data(self):
        if self._raw_data is None:
            self.read_comadre()
        return self._raw_data
