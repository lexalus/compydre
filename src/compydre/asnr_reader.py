import os
import re
import tempfile

from io import BytesIO

import networkx as nx
import requests

from bs4 import BeautifulSoup


ACCEPTED_CLASSES = [
    "Actinopterygii",
    "Amphibia",
    "Aves",
    "Insecta",
    "Mammalia",
    "Reptilia",
]
BASE_CLASS_URL = "https://github.com/bansallab/asnr/tree/master/Networks"

GITHUB_URL = "https://github.com"

BASE_RAW_GRAPHML = "https://raw.githubusercontent.com"


def urlopen(*args, **kwargs):
    """
    Lazy-import wrapper for stdlib urlopen, as that imports a big chunk of
    the stdlib.
    """
    import urllib.request

    return urllib.request.urlopen(*args, **kwargs)


class SpeciesGraph(object):
    def __init__(self, species, graph_link):
        self.species = species
        self._graph_link = graph_link
        self._graph = None

    def create_graph(self):
        url = BASE_RAW_GRAPHML + self._graph_link.replace("/blob", "")
        req = urlopen(url)
        reader = BytesIO(req.read())
        req.close()
        tmpdir = tempfile.mkdtemp()
        filename = "temp_{}.graphml".format(url.replace("/", "_"))
        path = os.path.join(tmpdir, filename)
        with open(path, "wb") as tmp:
            tmp.write(reader.getbuffer())
        g = nx.read_graphml(path)
        os.remove(path)
        os.rmdir(tmpdir)
        self._graph = g

    @property
    def graph(self):
        if self._graph is None:
            self.create_graph()
        return self._graph


class ASNRReader(object):
    def __init__(self):
        self._species_data = None
        self.base_url = "https://github.com/bansallab/asnr/tree/master/Networks"

    @staticmethod
    def create_metadata(link):
        metadata_url = GITHUB_URL + link
        resp = requests.get(metadata_url)
        soup = BeautifulSoup(resp.text, "html.parser")
        keys = [re.sub("<.*?>", "", str(i)) for i in soup.find_all("td")[2:][0:][::2]]
        values = [re.sub("<.*?>", "", str(i)) for i in soup.find_all("td")[2:][1:][::2]]
        metadata = dict(zip(keys, values))
        return metadata

    @staticmethod
    def find_links(url):
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.find_all("a", {"class": "js-navigation-open link-gray-dark"})
        return list(map(lambda x: x["href"], links))

    def create_species_graphs(self, species, link):
        url = GITHUB_URL + link
        links = self.find_links(url)[1:]
        species_graphs = list(map(lambda x: SpeciesGraph(species, x), links))
        return species_graphs

    def create_species_graph_dict(self, links, species_class):
        graph_dict = {}
        for link in links:
            value_dict = {}
            key = link.split("{}/".format(species_class))[1].split("_")[0]
            value_dict["graphs"] = self.create_species_graphs(key, link)
            value_dict["metadata"] = self.create_metadata(link)
            graph_dict[key] = value_dict
        return graph_dict

    def find_class_urls(self, species_class):
        url = self.base_url + "/" + species_class
        links = self.find_links(url)
        return links

    def create_species_data(self):
        species_data = {}
        print("Gathering data from the ASNR Repository")
        for species_class in ACCEPTED_CLASSES:
            class_urls = self.find_class_urls(species_class)
            graph_dict = self.create_species_graph_dict(class_urls, species_class)
            species_data[species_class] = graph_dict
        print("Data is now available")
        self._species_data = species_data
        return

    @property
    def species_data(self):
        if self._species_data is None:
            self.create_species_data()
        return self._species_data
