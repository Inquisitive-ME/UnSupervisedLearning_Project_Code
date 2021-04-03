# Code Location
github: 
Backup Google Drive:


# Data Sets
All the Data should be contained under the data folder in the main repo

The Noisy Non-Linear data set is generated using the functions in data/generated/generated_data.py
Analysis of the Noisy Non-Linear Data Set can be found in data/generated/Generated_Noisy_Nonlinear_Data_Analysis.ipynb

The pictures used in the faces data set can be found at https://susanqq.github.io/UTKFace/
The methods to generate the dataset can be found in data/faces/faces_generate_HOG_features.py
This will try to download the images if they do not already exist in data/faces/UTKFace
There is also a saved binary file containing the data in HOG_face_data.zip in order to avoid having to regenerate the data
Analysis of the Faces data set can be found in data/faces/faces_data_analysis.ipynb

# Code
All graphs in the report are saved directly from the jupyter notebooks in each respective folder,
therefore the graphs in the report may not be exactly what is reproduced in the jupyter notebook but should be close
and any differences should not affect the analysis provided in the report.

Each notebook uses several common python files which also exist in this repo.

The jupyter notebooks are organized into different folders based on the different parts for the project. The

## Part 1: Clustering Algorithms: K-means Expectation Maximization:

* Faces Data Set
    Clustering/faces_dataset_clustering.ipynb
* Noisy Non-Linear Data Set
    Clustering/generated_noisy_non-linear_clustering.ipynb

## Part 2: Dimensionality Reduction: PCA, ICA, RandomProjection, Boosting Feature Selection
* Faces Data Set
    DimensionalityReduction/Faces_Dimensionality_Reduction.ipynb
* Noisy Non-Linear Data Set
    DimensionalityReduction/generated_noisy_non-linear_Dimensionality_Reduction.ipynb

## Part 3: Clustering With Dimensionality Reduction:
* Faces Data Set
    Clustering_DimensionalityReduction/Faces_DR_Boosting_Clustering.ipynb 
    Clustering_DimensionalityReduction/Faces_DR_RandomProjection_Clustering.ipynb
    Clustering_DimensionalityReduction/Faces_DR_ICA_Clustering.ipynb
    Clustering_DimensionalityReduction/Faces_DR_PCA_Clustering.ipynb
* Noisy Non-Linear Data Set
    Clustering_DimensionalityReduction/generated_noisy_non-linear_DR_Boosted_Clustering.ipynb 
    Clustering_DimensionalityReduction/generated_noisy_non-linear_DR_Random_Clustering.ipynb
    Clustering_DimensionalityReduction/generated_noisy_non-linear_DR_ICA_Clustering.ipynb
    Clustering_DimensionalityReduction/generated_noisy_non-linear_DR_PCA_Clustering.ipynb

## Part 4: Neural Network with Dimensionality Reduction on Noisy Non-Linear Data Set
* Faces Data Set
    NN_DimensionalityReduction/faces_NN_analysis.ipynb
* Noisy Non-Linear Data Set
    NN_DimensionalityReduction/generated_noisy_nonlinear_NN_analysis_PCA.ipynb
    NN_DimensionalityReduction/generated_noisy_nonlinear_NN_analysis_ICA.ipynb
    NN_DimensionalityReduction/generated_noisy_nonlinear_NN_analysis_Random.ipynb
    NN_DimensionalityReduction/generated_noisy_nonlinear_NN_analysis_Boosted.ipynb

## Part 5: Neural Networks with Clustering Features on Noisy Non-Linear Data Set
* Noisy Non-Linear Data Set
    NN_Clustering/generated_noisy_nonlinear_SVM_analysis.ipynb

## Python packages
I think the python packages are pretty standard, but a requirements.txt file is provided in the repo listing all packages
that were installed when running the code

# References
The code primarily uses scikit-learn https://scikit-learn.org/stable/
And many examples from the scikit-learn docs https://scikit-learn.org/stable/auto_examples/index.html
All specific code references are in the code directly

UTKFace data set:
https://susanqq.github.io/UTKFace/

Class Lectures:
https://classroom.udacity.com/courses/ud262