import numpy as np

def CE(y):

    num_labels = y.shape[0]
    # case for low dimension y
    if num_labels <= 1:
        return 0

    # creates to arrays containing the values of y, and how many times they appear
    value, counts = np.unique(y, return_counts=True)
    # calculation of \pi(R_{\alpha})
    probs = counts / num_labels
    num_classes = np.count_nonzero(probs)

    # case if all the probabilities are zero
    if num_classes <= 1:
        return 0
    # initialise entropy value
    entropy = 0

    # compute the entropy
    for p in probs:
        entropy -= p*np.log(p)

    return entropy

# compute the information gain, after making a split
def information_gain(y, y1, y0):
    IG = CE(y) - (CE(y1)*y1.shape[0] + CE(y0)*y0.shape[0])/y.shape[0]
    return IG


# function which splits X and y based on the threshold
def split(X, y, feature_idx, threshold):
    # initialise as lists then convert to np arrays for efficiency
    X1, y1, X0, y0 = [], [], [], []
    # iterate through X and y, grouping values which satisfy and don't satisfy the condition
    for j in range(len(y)):
        if X[j][feature_idx] <= threshold:
            X1.append(X[j])
            y1.append(y[j])
        else:
            X0.append(X[j])
            y0.append(y[j])

    # convert to np arrays
    X1 = np.array(X1)
    y1 = np.array(y1)
    X0 = np.array(X0)
    y0 = np.array(y0)

    return X1, y1, X0, y0

# defines the best split at a node
def best_split(X, y, rs_idx):
    # initialising values
    num_features = X.shape[1]
    best_gain_cost = 0
    best_feature_idx = 0
    best_threshold = 0

    # iterate through the index of the random sample of features we choose, calculating the threshold and subsequent information gain
    for feature_idx in rs_idx:
        unique_vals = sorted(set(X[:, feature_idx]))

        for j in range(len(unique_vals) - 1):
            threshold = (unique_vals[j] + unique_vals[j+1])/2
            X1, y1, X0, y0 = split(X, y, feature_idx, threshold)
            info_gain_cost = information_gain(y, y1, y0)

            # update values if the new cost is the best so far
            if info_gain_cost > best_gain_cost:
                best_gain_cost = info_gain_cost
                best_feature_idx = feature_idx
                best_threshold = threshold

    return best_feature_idx, best_threshold