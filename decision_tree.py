import numpy as np
from scipy.stats import mode
from utilities import CE, information_gain, split, best_split

class Node(object):
    # initialise the object
    def __init__(self, feature_idx, threshold, branch1, branch0):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.branch1 = branch1
        self.branch0 = branch0


class decisiontree(object):
    # initialise the attributes of the decision tree class
    # num_predictors defines the number of predictors used at each split
    # maximum depth of the tree
    # minimum samples needed at a given node, to determine a new split, almost always I will set this to 1
    def __init__(self, num_predictors, max_depth,
                    min_samples_leaf):
        self.num_predictors = num_predictors
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf


    # function which builds the tree recursively, very similar to that in the CT
    def build_tree(self, X, y, rs_idx, depth):
        
        # clause for no features, or depth is too great, or X is too small, or cross entropy is zero, or all labels are the same
        if len(rs_idx)==0 or depth >= self.max_depth or len(X) < self.min_samples_leaf or CE(y) == 0 or len(np.unique(y))==1:
            return mode(y)[0][0]

        # compute the best split
        feature_idx, threshold = best_split(X, y, rs_idx)

        # split the data and added clause for zero dimensional y
        X1, y1, X0, y0 = split(X, y, feature_idx, threshold)
        if y1.shape[0] == 0 or y0.shape[0] == 0:
            return mode(y)[0][0]
        
        # create the branches recursively
        branch1 = self.build_tree(X1, y1, rs_idx, depth+1)
        branch0 = self.build_tree(X0, y0, rs_idx, depth+1)

        return Node(feature_idx, threshold, branch1, branch0)

    # train function which constructs the tree
    def train(self, X, y):

        num_features = X.shape[1]
        # governs number of predictors considered at each split
        num_sub_features = int(self.num_predictors(num_features))
        # index of the random sample of features we select
        rs_idx = random.sample(range(num_features), num_sub_features)
        
        self.treebuild = self.build_tree(X, y, rs_idx, 0)

    # function which predicts of each instance in X
    def predict(self, X):
        # initialisation
        num_samples = X.shape[0]
        class_pred = np.zeros(num_samples)
        # iterate through the samples, constructing a node and checking if our features satisfy the threshold condition
        for j in range(num_samples):
            node = self.treebuild

            while isinstance(node, Node):
                if X[j][node.feature_idx] <= node.threshold:
                    node = node.branch1
                else:
                    node = node.branch0
            class_pred[j] = node

        return class_pred