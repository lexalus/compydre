# Compydre 
## A Python package for manipulating the Comadre Database

The Comadre Animal Matrix Database contains matrix population models (MPM) 
for 417 different taxonomic species. These MPMs can be used to project growth of a population
and for visualizing life cycle graphs. Unfortunately, the analytical support for this database is exclusively in R.
This package is designed to easily import a local Comadre file and create a data structure for quick data retrieval
in Python.

Compydre 0.0.1 consists of two different classes: ComadreTransformer and SpeciesMatrix.

### ComadreTransformer

To use the ComadreTransformer, you must first download the RData from [Comadre|https://compadre-db.org/Data/Comadre].
Once it is downloaded, rename the file to "comadre.RData" and take note of your absolute path.

You initialize the ComadreTransformer like this:

ct = ComadreTransformer("my/absolute/path/comadre.RData")

To retrieve the data, you simply call the attribute "species_data":

data = ct.species_data

This will return a dictionary with the Comadre data. The keys are the species and the values are lists containing
SpeciesMatrix objects. Each object represents one unique study for the species reported in Comadre.

### SpeciesMatrix

The SpeciesMatrix object contains unique study data retrieved from Comadre.

#### SpeciesMatrix.species
This is the species associated with the study.

#### SpeciesMatrix.mat_a
The matrix reflecting the population dynamics of the species population observed.

#### SpeciesMatrix.mat_u
The population sub-matrix which describes the survival-dependent dynamics of the species population observed.

#### SpeciesMatrix.mat_f
The population sub-matrix which describes the sexual-reproduction dynamics of the species population observed.

#### SpeciesMatrix.mat_c
The population sub-matrix which describes the clonal-reproduction dynamics of the species population observed.

#### SpeciesMatrix.metadata
This is the metadata of the study as recorded by Comadre. You can review all possible metadata features in their user guide 
https://github.com/jonesor/compadreDB/blob/master/COMADRE-UserGuide/COMADRE-UserGuide.pdf.

#### SpeciesMatrix.index
This is the index of the study within the Comadre database.

#### SpeciesMatrix.study_info
This is a DataFrame detailing the population types observed in the study. It includes one column named StudyPopulationGroup
and MatrixId. StudyPopulationGroup represents the characteristic (usually age) of the observed group. MatrixId reflects the 
id of the matrix in Comadre. The DataFrame is sorted to reflect the MPM rows/columns, where SpeciesMatrix.study_info.iloc[0]
represents the MPM row and column index of 0.

#### SpeciesMatrix.lc_graph
This is a networkx weighted graph that represents the life cycle dynamics of the species population as detailed in Species.Matrix.mat_a.