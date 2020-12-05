import pytest

from compydre import asnr_reader


def test_species_graph():
    species_graph = asnr_reader.SpeciesGraph("Gorilla", "/path/to/bananas")

    assert species_graph.species == "Gorilla"
    assert species_graph._graph_link == "/path/to/bananas"
    assert species_graph._graph is None


# TODO: FINISH TESTS
# def test_species_create_graph(mocker):
#     mock_url_open = mocker.patch("compydre.asnr_reader.urlopen")
#     mock_bytes_io = mocker.patch("compydre.asnr_reader.BytesIO")
#     mock_tempfile = mocker.patch("compydre.asnr_reader.tempfile")
#     mock_os = mocker.patch("compydre.asnr_reader.os")
#
#     species_graph = asnr_reader.SpeciesGraph("Gorilla", "www.bananas.com")
