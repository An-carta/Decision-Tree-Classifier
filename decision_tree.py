import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split                     # import function to split dataset
from sklearn.metrics import accuracy_score                               # import function to compute accuracy                     
from sklearn.tree import DecisionTreeClassifier
import time                                                              


class Node:
    """
    Node class for decision tree representation.

    Attributes:
    - feature: Feature index for splitting.
    - threshold: Threshold for the feature split.
    - left: Left branch (subtree).
    - right: Right branch (subtree).
    - value: Value for leaf nodes, represents the predicted class.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature                    # feature on which the node will be splitted
        self.threshold = threshold                # threshold of the split 
        self.left = left                          # left branch
        self.right = right                        # right branch
        self.value = value                        # value of the node in case it's a leaf

    def is_leaf_node(self):
        """
        Check if the node is a leaf.

        Returns:
        - True if the node is a leaf, False otherwise.
        """
        return self.value is not None                        # check whether a value is assigned to the node


class DecisionTree:
    """
    DecisionTree class for building and using a decision tree classifier.

    Attributes:
    - root: Root node of the decision tree.

    Methods:
    - learn(X_train, y_train, impurity_measure='entropy', prune=False, pruning_proportion=0.2): Fit the decision tree to the training data.
    - predict(X): Predict labels for a given dataset.
    - _grow_tree(X, y, impurity_measure): Recursively grow the decision tree.
    - _best_split(X, y, feats_idxs, impurity_measure): Find the best feature and threshold to split on.
    - _information_gain(y, X_column, threshold): Calculate information gain for a split.
    - _split(X_column, split_thresh): Split a column over a given threshold.
    - _entropy(y): Calculate the entropy of a set of labels.
    - _traverse_tree(x, node): Traverse the decision tree for a single data point.
    - _gini(y): Calculate the Gini index for a set of labels.
    - _gini_gain(y, X_column, threshold): Calculate Gini gain for a split.
    - _prune_tree(X_prune, y_prune): Post-prune the decision tree.
    - _bottom_up_prune(node, X_prune, y_prune): Bottom-up pruning of the decision tree.
    - _calc_accuracy(X, y): Calculate accuracy for a dataset.

    """

    def __init__(self):
        self.root = None

    def learn(self, X_train, y_train, impurity_measure='entropy', prune=False, pruning_proportion = 0.2):
        """
        Fit the decision tree to the training data.

        Args:
        - X_train: Training feature matrix.
        - y_train: Training labels.
        - impurity_measure: Impurity measure ('entropy' or 'gini').
        - prune: Whether to perform post-pruning.
        - pruning_proportion: Proportion of the pruning set

        Returns:
        - None
        """
        if prune:                                                                                                           # possibility to post-prune the decision tree
            X_train, X_prune, y_train, y_prune = train_test_split(X_train, y_train, test_size = pruning_proportion)         # split the training data
            self.root = self._grow_tree(X_train, y_train, impurity_measure)                                                 # start growing tree
            self._prune_tree(X_prune, y_prune)                                                                              # post prune the tree just built
        else:   
            self.root = self._grow_tree(X_train, y_train, impurity_measure)                                                 # start building the tree on the full training data

    def _grow_tree(self, X, y, impurity_measure):
        """
        Recursively grow the decision tree.

        Args:
        - X: Feature matrix.
        - y: Labels.
        - impurity_measure: Impurity measure ('entropy' or 'gini').

        Returns:
        - Root node of the decision tree.
        """
        n_samples, n_feats = X.shape                                                               # get the number of samples and features 
        n_labels = len(np.unique(y))                                                               # get the number of labels

        # stopping criteria
        if n_labels == 1:                                                                          # all the samples have the same label
            leaf_value = y[0]
            return Node(value=leaf_value)                                                          # return a leaf with that value
        elif all(np.all(X[:, i] == X[0, i]) for i in range(n_feats)):                              # all the samples have the same value of each feature 
            leaf_value = np.argmax(np.bincount(y))                                                 # bincount returns the number of occurrences of the index
            return Node(value=leaf_value)                                                          # return a leaf with value the majority label
        # recursive step 
        else:
            feats_idxs = range(n_feats)                                                            # get the indices of the features
            best_feature, best_thresh = self._best_split(X, y, feats_idxs, impurity_measure)       # get the best feature and the best threshold to split on

            left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)                   # split the dataset in two

            left = self._grow_tree(X[left_idxs, :], y[left_idxs], impurity_measure)                # recursively build a tree on the left and on the right
            right = self._grow_tree(X[right_idxs, :], y[right_idxs], impurity_measure)
            

            return Node(best_feature, best_thresh, left, right)                                    # return a node, that is not a leaf

    def _best_split(self, X, y, feats_idxs, impurity_measure):
        """
        Find the best feature and threshold to split on.

        Args:
        - X: Feature matrix.
        - y: Labels.
        - feats_idxs: Indices of features to consider.
        - impurity_measure: Impurity measure ('entropy' or 'gini').

        Returns:
        - split_idx: Index of the best feature.
        - split_threshold: Threshold for the best split.
        """
        best_gain = -1                                                         # initialize the best gain to a negative number
        split_idx, split_threshold = None, None                                # initialize the variables to return to None

        for feat_idx in feats_idxs:                                            # iterate through all the features
            X_column = X[:, feat_idx]                                          # get the current column
            thresholds = np.unique(X_column)                                   # get all the values of that feature

            for thr in thresholds:                                             # iterate through all the values of that feature
                if impurity_measure == 'entropy':                              
                    gain = self._information_gain(y, X_column, thr)            # compute information gain for each threshold 
                elif impurity_measure == 'gini':
                    gain = self._gini_gain(y, X_column, thr)                   # compute gini gain for each threshold

                if gain > best_gain:                                           # if the current gain is better than the best one update it
                    best_gain = gain
                    split_idx = feat_idx                                       # keep track of the current feature and threshold
                    split_threshold = thr
                    
        return split_idx, split_threshold                                      # return the best feature and the best threshold to split on

    def _information_gain(self, y, X_column, threshold):
        """
        Calculate information gain for a split.

        Args:
        - y: Labels.
        - X_column: Feature column to split.
        - threshold: Threshold value for the split.

        Returns:
        - information_gain: Information gain.
        """
        parent_entropy = self._entropy(y)                                       # get the parent entropy

        left_idxs, right_idxs = self._split(X_column, threshold)                # split over a threshold
        if len(left_idxs) == 0 or len(right_idxs) == 0:                         # if there is no spit there is no information gain
            return 0

        n = len(y)                                                              # get total number of labels
        n_l, n_r = len(left_idxs), len(right_idxs)                              # get weights of the branches
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])    # get entropy of the branches

        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r                       # compute the conditional entropy as weighted average of the entropies of the branches

        information_gain = parent_entropy - child_entropy                       # compute information gain

        return information_gain

    def _split(self, X_column, split_thresh):
        """
        Split a column over a given threshold.

        Args:
        - X_column: Feature column to split.
        - split_thresh: Threshold value for the split.

        Returns:
        - left_idxs: Indices of samples in the left branch.
        - right_idxs: Indices of samples in the right branch.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()             # divide left and right based on the threshold
        right_idxs = np.argwhere(X_column > split_thresh).flatten()

        return left_idxs, right_idxs

    def _entropy(self, y):
        """
        Calculate the entropy of a set of labels.

        Args:
        - y: Labels.

        Returns:
        - entropy: Entropy value.
        """
        hist = np.bincount(y)                           # bincount returns a list filled with the number of occurences of each label
        ps = hist / len(y)                              # compute the probability of each label
        entropy = 0                             
        for p in ps:                                    # iterate over all probabilities
            if p > 0:
                entropy += -(p * np.log2(p))            # compute entropy

        return entropy

    def predict(self, X):
        """
        Predict labels for a given dataset.

        Args:
        - X: Feature matrix of the dataset.

        Returns:
        - predictions: Predicted labels.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])       # traverse the tree for every piece of data

    def _traverse_tree(self, x, node):
        """
        Traverse the decision tree for a single data point.

        Args:
        - x: Feature vector for a single data point.
        - node: Current node being evaluated.

        Returns:
        - Node value (predicted label).
        """
        # base case
        if node.is_leaf_node():                            
            return node.value                              # return leaf value
        
        # recursive step
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)       # recursively traverse left and right
        else:
            return self._traverse_tree(x, node.right)

    def _gini(self, y):
        """
        Calculate the Gini index for a set of labels.

        Args:
        - y: Labels.

        Returns:
        - gini: Gini index.
        """
        hist = np.bincount(y)                              # bincount returns the number of occurrences of each label
        ps = hist / len(y)                                 # compute the probability of each label 
        
        gini = np.sum(ps * (1 - ps))                       # compute the gini index
        
        return gini

    def _gini_gain(self, y, X_column, threshold):
        """
        Calculate Gini gain for a split.

        Args:
        - y: Labels.
        - X_column: Feature column to split.
        - threshold: Threshold value for the split.

        Returns:
        - gini_gain: Gini gain.
        """
        left_idxs, right_idxs = self._split(X_column, threshold)              # split over the threshold 
        if len(left_idxs) == 0 or len(right_idxs) == 0:                       # no split-> no gain
            return 0

        n = len(y)                                                            # get number of labels
        n_l, n_r = len(left_idxs), len(right_idxs)                            # get lenght of the branches
        g_l, g_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])        # get gini index of the branches
        
        gini_parent = self._gini(y)                                           # get the gini index of the node
        gini_child = ((n_l / n) * g_l + (n_r / n) * g_r)                      # compute conditional gini index
        
        gini_gain = gini_parent - gini_child                                  # compute gini gain

        return gini_gain

    def _prune_tree(self, X_prune, y_prune):
        """
        Post-prune the decision tree.

        Args:
        - X_prune: Feature matrix for pruning.
        - y_prune: Labels for pruning.

        Returns:
        - None
        """
        self._bottom_up_prune(self.root, X_prune, y_prune)

    def _bottom_up_prune(self, node, X_prune, y_prune):
        """
        Bottom-up pruning of the decision tree.

        Args:
        - node: Current node being evaluated.
        - X_prune: Feature matrix for pruning.
        - y_prune: Labels for pruning.

        Returns:
        - None
        """
        # base case
        if node.is_leaf_node():                                              # no need to prune a leaf
            
            return

        # recursive step
        if node.left:                                                        
            self._bottom_up_prune(node.left, X_prune, y_prune)              # prune left and right subtrees
        if node.right:
            self._bottom_up_prune(node.right, X_prune, y_prune)

        
        accuracy_before = self._calc_accuracy(X_prune, y_prune)             # compute accuracy before pruning

        
        original_value = node.value                                         # try pruning the current subtree
        node.value = np.argmax(np.bincount(y_prune))                        # the index of the bincount max value corresponds to the the value havind the max number of occurences

        
        accuracy_after = self._calc_accuracy(X_prune, y_prune)              # compute accuracy after pruning

        
        if accuracy_after >= accuracy_before:                               # check if pruning doesn't decrease accuracy
            return                                                          # Keep the pruned subtree

        
        node.value = original_value                                         # if pruning decreases accuracy, restore the original value

    def _calc_accuracy(self, X, y):
        """
        Calculate accuracy for a dataset.

        Args:
        - X: Feature matrix of the dataset.
        - y: True labels of the dataset.

        Returns:
        - accuracy: Accuracy value.
        """
        
        predictions = self.predict(X)                                        # make predictions using the decision tree
        
        
        correct_predictions = np.sum(predictions == y)                       # calculate accuracy by comparing predictions to true labels
        total_samples = len(y)
        accuracy = correct_predictions / total_samples

        return accuracy





data = pd.read_csv("wine_dataset.csv")                                                           # open file


if isinstance(data, pd.DataFrame):                                                               # if the dataset is stored in a pandas dataframe turn it into numpy array
    data = data.to_numpy()
    X, y = data[:, :-1], data[:, -1]                                                             # split samples from labels          
    y = y.astype(int)                                                                            # consider the labels as ints since we are dealing with classification
else:
    X, y = data.data, data.target
    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)                       # split dataset in training and test sets
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5)                 # split the test set in validation and test set

my_models = [                                                                                    # list of models with different hyperparameters
    {'name': 'Model 1 - Entropy (No Pruning)', 'impurity_measure': 'entropy', 'prune': False},
    {'name': 'Model 2 - Entropy (Pruning)', 'impurity_measure': 'entropy', 'prune': True},
    {'name': 'Model 3 - Gini (No Pruning)', 'impurity_measure': 'gini', 'prune': False},
    {'name': 'Model 4 - Gini (Pruning)', 'impurity_measure': 'gini', 'prune': True},
]


best_accuracy = 0                  # initialise accuracy to 0
best_model = None                  # initialise best model to none

for model in my_models:                                                                            # iterate over all the models
        
    clf = DecisionTree()                                                                           # initialize the DecisionTree model
    clf.learn(X_train, y_train, impurity_measure=model['impurity_measure'], prune=model['prune'])  # fit the model to the training data
    y_pred = clf.predict(X_val)                                                                    # predict validation data

    accuracy = accuracy_score(y_val, y_pred)                                                       # get accuracy on validation set                           
        
    print(f"{model['name']} - Validation Accuracy: {accuracy:.4f}")
    
    if accuracy > best_accuracy:                                                                   # update the best accuracy and the best model
        best_accuracy = accuracy
        best_model = model
        
print(f"\nBest model: {best_model['name']}, Validation Accuracy: {best_accuracy}.\n")


my_clf = DecisionTree()                                                                                   # initialize the model selected on validation data
my_start_time = time.time()                                                                               # get start time
clf.learn(X_train, y_train, impurity_measure=best_model['impurity_measure'], prune=best_model['prune'])     # fit using best hyperparameters 
my_end_time = time.time()                                                                                 # get finishing time

y_pred = clf.predict(X_test)                                                                              # predict on the test set

test_accuracy = accuracy_score(y_test, y_pred)                                                            # get accuracy on the test set

print(f"Best model: {best_model['name']}, Test Accuracy: {test_accuracy}.")
print(f"My Decision Tree Training Time: {my_end_time - my_start_time:.4f} seconds\n")


sklearn_model = DecisionTreeClassifier(criterion=best_model['impurity_measure'])                          # initialise sklearn model with same impurity measure as my best model          
sklearn_start_time = time.time()                                                                          # get starting time
sklearn_model.fit(X_train, y_train)                                                                       # fit to training data
sklearn_end_time = time.time()                                                                            # get finishing time

sklearn_predictions = sklearn_model.predict(X_test)                                                       # predict test set

sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)                                            # get accuracy on test set

print(f"scikit-learn DecisionTreeClassifier Accuracy: {sklearn_accuracy:.4f}")
print(f"scikit-learn DecisionTreeClassifier Training Time: {sklearn_end_time - sklearn_start_time:.4f} seconds")