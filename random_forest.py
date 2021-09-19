import numpy as np
from scipy.stats import mode
from decision_tree import *
from utilities import CE, information_gain, split, best_split
import pandas as pd

class randomforest(object):
    # initialise the attributes of the class
    # same as before apart from num_trees= number of trees in the forest
    # bootstrap, a variable in (0,1) from which we consider (number of samples)*(boostrap value) within our tree - this helps reduce overfitting
    def __init__(self, num_trees, num_predictors, max_depth,
        min_samples_leaf, bootstrap):

        self.num_trees = num_trees
        self.num_predictors = num_predictors
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.forest = []

    # training function for the forest
    def train(self, X, y):
        # initialising variables and creating the number of sub samples as mentioned
        self.forest = []
        num_samples = X.shape[0]
        num_sub_samples = int(round(num_samples*self.bootstrap))
        
        # iterate through the number of trees, each time building a tree and appending this to the forest for which we will later use to predict
        for i in range(self.num_trees):
            # generate reduced X_train and y_train
            X_subset = X[:num_sub_samples]
            y_subset = y[:num_sub_samples]

            # build the tree, calling the decisiontree object
            tree = decisiontree(self.num_predictors, self.max_depth, self.min_samples_leaf)
            tree.train(X_subset, y_subset)
            self.forest.append(tree)

    # make predictions from the forest 
    def predict(self, X):
        num_samples = X.shape[0]
        num_trees = self.num_trees
        # prediction matrix to store values
        preds = np.zeros([num_trees, num_samples])
        for i in range(num_trees):
            preds[i,:] = self.forest[i].predict(X)

        return mode(preds)[0][0]

    # score function to determine accuracy or the confusion matrix
    def score(self, X_test, y_test, confusion=False):
        # y prediction
        y_pred = self.predict(X_test)
        if confusion==False:
            # accuracy
            return np.float(sum(y_pred==y_test))/float(len(y_test))
        else:
            # confusion matrix
            y_pred = pd.Series(y_pred, name='Predicted')
            y_true = pd.Series(y_test, name='Actual')
            confusion_mat = pd.crosstab(y_true, y_pred)
            return confusion_mat


