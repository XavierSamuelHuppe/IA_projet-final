import numpy as np
import matplotlib.pyplot as plt
import sys
import load_datasets
import NeuralNet  # importer la classe du Réseau de Neurones
import DecisionTree  # importer la classe de l'Arbre de Décision
# importer d'autres fichiers et classes si vous en avez développés
# importer d'autres bibliothèques au besoin, sauf celles qui font du machine learning

train_pourcentage_iris = 0.7
train_pourcentage_congressional = 0.7
k_folds = 10 #number of folds

dataset_iris = load_datasets.load_iris_dataset(train_pourcentage_iris)
dataset_congressional = load_datasets.load_congressional_dataset(train_pourcentage_congressional)
dataset_monks_1 = load_datasets.load_monks_dataset(1)
dataset_monks_2 = load_datasets.load_monks_dataset(2)
dataset_monks_3 = load_datasets.load_monks_dataset(3)


neural_network_iris = NeuralNet.NeuralNet()

cross_validation_iris = neural_network_iris.cross_validation(dataset_iris[0], dataset_iris[1], 10)
