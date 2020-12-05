import pytest
import numpy as np
import pandas as pd
import networkx as nx

from compydre import comadre_reader


@pytest.fixture
def mock_species_data(mock_species_matrix):

    test_species_data = {
        "species1": [mock_species_matrix, mock_species_matrix, mock_species_matrix],
        "species2": [mock_species_matrix, mock_species_matrix],
        "species3": [mock_species_matrix],
    }

    return test_species_data


@pytest.fixture
def mock_matrices():
    return {
        "matA": np.array([[0.0, 0.0, 1.12], [0.342, 0.0, 0.0], [0.0, 0.951, 0.948]]),
        "matU": np.zeros([2, 2]),
        "matF": np.zeros([2, 2]),
        "matC": np.zeros([2, 2]),
    }


@pytest.fixture
def mock_species_matrix(mock_matrices, mock_metadata):
    return comadre_reader.SpeciesMatrix(
        "species1", mock_matrices, None, mock_metadata, 0
    )


@pytest.fixture
def mock_study_info():
    arr = np.array(
        [
            ["Calf: 0-1 years", "240296"],
            ["Yearling: 1-2 years", "240296"],
            ["Adult: 2+ years", "240296"],
        ]
    )
    df = pd.DataFrame(arr, columns=["StudyPopulationGroup", "MatrixId"])
    return df


@pytest.fixture
def mock_metadata():
    arr = np.array(
        [["This is a mock", "I am mocking", "Don't mock the boat"] for i in range(0, 6)]
    )
    df = pd.DataFrame(arr, columns=["metadata1", "metadata2", "metadata3"])
    return df


@pytest.fixture
def mock_lc_graph(mock_study_info, mock_matrices):
    G = nx.Graph()

    edge_list = [
        ("Adult: 2+ years", "Calf: 0-1 years", 1.12),
        ("Calf: 0-1 years", "Yearling: 1-2 years", 0.342),
        ("Yearling: 1-2 years", "Adult: 2+ years", 0.951),
        ("Adult: 2+ years", "Adult: 2+ years", 0.948),
    ]

    G.add_weighted_edges_from(edge_list)
    return G


def test_species_matrix(mock_matrices, mock_study_info, mock_metadata):
    test_species = "Gorilla gorilla"
    test_index = 0

    actual = comadre_reader.SpeciesMatrix(
        test_species, mock_matrices, mock_study_info, mock_metadata, test_index
    )

    assert test_species == actual.species
    assert test_index == actual.index
    assert actual._lc_graph is None

    pd.testing.assert_frame_equal(actual.study_info, mock_study_info)
    pd.testing.assert_frame_equal(actual.metadata, mock_metadata)
    np.testing.assert_array_equal(actual.mat_a, mock_matrices["matA"])
    np.testing.assert_array_equal(actual.mat_u, mock_matrices["matU"])
    np.testing.assert_array_equal(actual.mat_f, mock_matrices["matF"])
    np.testing.assert_array_equal(actual.mat_c, mock_matrices["matC"])


def test_project_species_growth(mock_matrices, mock_study_info, mock_metadata):
    test_species = "Gorilla gorilla"
    test_index = 0

    expected = {
        0: np.array([100, 100, 100]),
        1: np.array([112.0, 34.2, 189.9]),
        2: np.array([212.688, 38.304, 212.5494]),
    }

    test_class = comadre_reader.SpeciesMatrix(
        test_species, mock_matrices, mock_study_info, mock_metadata, test_index
    )

    actual = test_class.project_species_growth(2)

    for i in range(0, 3):
        np.testing.assert_array_almost_equal(actual[i], expected[i])


def test_return_lc_graph(mock_matrices, mock_study_info, mock_lc_graph, mock_metadata):
    expected = [(u, v, d) for (u, v, d) in mock_lc_graph.edges(data=True)]
    test_species = "Gorilla gorilla"
    test_index = 0

    test_class = comadre_reader.SpeciesMatrix(
        test_species, mock_matrices, mock_study_info, mock_metadata, test_index
    )
    actual = [(u, v, d) for (u, v, d) in test_class.return_lc_graph().edges(data=True)]

    assert actual == expected


def test_comadre_reader():

    test_class = comadre_reader.ComadreReader("path")

    assert test_class.path == "path"
    assert test_class._raw_data is None


def test_comadre_read(mocker):
    mock_r = mocker.patch("compydre.comadre_reader.r")
    mock_comadre = mocker.patch("compydre.comadre_reader.r.comadre", 2)
    test_class = comadre_reader.ComadreReader("path")

    actual = test_class.read_comadre()

    mock_r.load.assert_called_once_with(test_class.path)

    assert test_class._raw_data == mock_comadre
    assert actual is None


def test_create_species_dict(
    mocker, mock_matrices, mock_species_matrix, mock_species_data, mock_metadata
):

    test_species_matrix = mock_species_matrix
    expected = mock_species_data

    test_class = comadre_reader.ComadreReader("path")

    mock_raw_data = mocker.patch.object(test_class, "_raw_data")
    mock_raw_matrices = mocker.patch.object(test_class, "return_matrices")
    mock_study_info = mocker.patch.object(test_class, "return_study_info")
    mock_metadataz = mocker.patch.object(test_class, "return_metadata")

    mock_species_matrix = mocker.patch.object(comadre_reader, "SpeciesMatrix")
    mock_species_matrix.return_value = test_species_matrix

    mock_raw_matrices.return_value = mock_matrices
    mock_study_info.return_value = None
    mock_metadataz.return_value = mock_metadata
    mock_rx2_call = mocker.Mock()

    mock_raw_data.rx2.return_value = mock_rx2_call
    mock_rx2_call.rx2.return_value = [
        "species1",
        "species1",
        "species1",
        "species2",
        "species2",
        "species3",
    ]

    actual = test_class.create_species_dict()

    assert expected == actual


def test_return_matrices(mocker):
    test_class = comadre_reader.ComadreReader("path")
    mock_raw_data = mocker.patch.object(test_class, "_raw_data")
    mock_raw_data.rx2.return_value = [
        [[0, 1, 2], [[0, 0], [0, 0]], [7, 8, 9]],
        [[10, 11, 12], [14, 15, 16], [17, 18, 19]],
    ]

    expected = np.array([[0, 0], [0, 0]])
    actual = test_class.return_matrices(0, 1)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("exists", (True, False))
def test_return_species(monkeypatch, mock_species_data, exists):
    monkeypatch.setattr(comadre_reader.ComadreReader, "species_data", mock_species_data)

    test_class = comadre_reader.ComadreReader("path")

    if exists:
        actual = test_class.return_species("species1")
        expected = mock_species_data["species1"]
        assert actual == expected
    else:
        with pytest.raises(Exception):
            test_class.return_species("species")
