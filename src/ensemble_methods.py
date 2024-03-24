import numpy as np
from scipy import stats

# If you want to use your own Decision Tree implementation for Random Forest
from decision_tree import DTClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict

# Something useful for tracking algorithm's iterations
import progressbar

widgets = ['Model Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),
           ' ', progressbar.ETA()]


def get_bootstrap_samples(X, y, nr_bootstraps, nr_samples=None):
    # region Summary
    """
    This function is for getting bootstrap samples with replacement from the initial dataset (X, y).
    Hint: you may need np.random.choice function somewhere in this function.
    :param X: Dataset
    :param y: 
    :param nr_bootstraps: The number of bootstraps needed
    :param nr_samples: The number of data points to sample each time. It should be the size of X, if nr_samples is not provided
    :return: Bootstrap samples
    """
    # endregion Summary

    # region Body

    if nr_samples is None:
        nr_samples = np.shape(X)[0]

    bootstrap_samples = []
    for i in range(nr_bootstraps):
        # region Variant N

        idx = np.random.choice(range(np.shape(X)[0]), size=nr_samples, replace=True)
        bootstrap_samples.append([X[idx, :], y[idx]])

        # endregion Variant N

        # region Variant A

        # l = nr_samples if nr_samples is not None else len(X)
        # indices = np.random.randint(0, len(X), size=l)
        # bootstrap_samples.append((X[indices], y[indices]))

        # endregion Variant A

    return bootstrap_samples

    # endregion Body


class Bagging:
    # region Constructor

    def __init__(self, base_estimator, nr_estimators=10):
        # region Summary
        """
        Constructor of Bagging class.
        :param base_estimator: Any object that has fit() and predict() methods
        :param nr_estimators: Number of models in the ensemble
        """
        # endregion Summary

        self.base_estimator = base_estimator
        self.nr_estimators = nr_estimators
        self.progressbar = progressbar.ProgressBar(widgets=widgets)

    # endregion Constructor

    # region Functions

    def fit(self, X, y):
        # region Summary
        """
        This method will fit a separate model (self.base_estimator) on each bootstrap sample and each model should be
        stored in order to use it in predict() method.
        :param X: Dataset
        :param y:
        """
        # endregion Summary

        # region Body

        X = np.array(X)
        y = np.array(y)
        bootstrap_samples = get_bootstrap_samples(X, y, nr_bootstraps=self.nr_estimators)
        self.models = []
        for i in self.progressbar(range(self.nr_estimators)):
            model = self.base_estimator()
            X_boot, y_boot = bootstrap_samples[i]
            model.fit(X_boot, y_boot)
            self.models.append(model)

        # endregion Body

    def predict(self, X):
        # region Summary
        """
        This method will predict the labels for a given test dataset.
        Get the majority 'vote' for each test instance from the ensemble.
        Hint: you may want to use mode() method from 'scipy.stats'.
        :param X: Dataset
        :return: Predicted labels for a given test dataset
        """
        # endregion Summary

        # region Body

        # region Variant N

        X = np.array(X)
        nr_estimators = self.nr_estimators
        y_predictions = np.zeros((X.shape[0], nr_estimators))
        for i in range(nr_estimators):
            y_predictions[:, i] = self.models[i].predict(X)
        return stats.mode(y_predictions, axis=1)[0]

        # endregion Variant N

        # region Variant A

        # votes = []
        # for i in self.models:
        #     votes.append(i.predict(X))
        # y_predictions = stats.mode(votes, axis=0)[0][0]
        # return y_predictions

        # endregion Variant A

        # endregion Body

    # endregion Functions


class RandomForest:
    # region Constructor

    def __init__(self, nr_estimators=10, max_features=None, min_samples_split=2, min_gain=0, max_depth=float("inf")):
        # region Summary
        """
        Constructor of RandomForest class.
        :param nr_estimators: Number of trees in the forest
        :param max_features: Number of features to use for each tree. If not specified, this should be set to sqrt(initial number of features)
        :param min_samples_split: Minimum samples split for decision tree
        :param min_gain: Minimum information gain of decision tree
        :param max_depth: Maximum depth of decision tree
        """
        # endregion Summary

        self.nr_estimators = nr_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.progressbar = progressbar.ProgressBar(widgets=widgets)

    # endregion Constructor

    # region Functions

    def fit(self, X, y):
        # region Summary
        """
        This method will fit a separate tree on each bootstrap sample and subset of features each tree should be stored
        in order to use it in predict() method.
        :param X: Dataset
        :param y:
        """
        # endregion Summary

        # region Body

        X = np.array(X)
        y = np.array(y)
        nr_features = np.shape(X)[1]
        if not self.max_features:
            self.max_features = int(np.sqrt(nr_features))

        bootstrap_samples = get_bootstrap_samples(X, y, self.nr_estimators)

        self.trees = []
        for i in self.progressbar(range(self.nr_estimators)):
            # It is possible to use either sklearn, or custom implemented Decision Tree with respective parameters
            # Custom Decision Tree:
            tree = DTClassifier(
                min_samples_split=self.min_samples_split,
                min_impurity=self.min_gain,
                max_depth=self.max_depth)
            X_boot, y_boot = bootstrap_samples[i]

            # region Variant N

            idx = np.random.choice(range(nr_features), size=self.max_features, replace=False)

            # endregion Variant N

            # region Variant A

            # size = self.max_features if self.max_features is not None else int(np.sqrt(len(X[0])))
            # idx = np.random.choice(np.arange(0, len(X[1])), size=size)

            # endregion Variant A

            # We need to keep the indices of the features used for this tree
            tree.feature_indices = idx
            tree.fit(X_boot[:, idx], y_boot)
            self.trees.append(tree)

        # endregion Body

    def predict(self, X):
        # region Summary
        """
        This method will predict the labels for a given test dataset.
        Get the majority vote for each test instance from the forest.
        Hint: you may want to use mode() method from 'scipy.stats'.
        Besides the individual trees, you will also need the feature indices it was trained on.
        :param X: Dataset
        :return: The labels for a given test dataset
        """
        # endregion Summary

        # region Body

        # region Variant N

        X = np.array(X)
        nr_estimators = self.nr_estimators
        y_predictions = np.zeros((X.shape[0], nr_estimators))
        for i, tree in enumerate(self.trees):
            idx = tree.feature_indices
            y_predictions[:, i] = tree.predict(X[:, idx])
        return stats.mode(y_predictions, axis=1)[0]

        # endregion Variant N

        # region Variant A

        # votes = []
        # for i in self.trees:
        #     votes.append(i.predict(X[:, i.feature_indices]))
        # y_predictions = stats.mode(votes, axis=0)[0][0]
        # return y_predictions

        # endregion Variant A

        # endregion Body

    # endregion Functions


class WeightedVoting:
    # region Constructor

    def __init__(self, estimators, k=5):
        # region Summary
        """
        Constructor of WightedVoting class.
        :param estimators: List of classifier objects
        :param k: Count of folds of Cross-Validation
        """
        # endregion Summary

        self.estimators = estimators
        self.nr_estimators = len(estimators)
        self.weights = None
        self.k = k

    # endregion Constructor

    # region Functions

    def get_weights(self, X, y):
        # region Summary
        """
        This method is for deriving the weights of each individual classifier using cross-validation as described in the lecture slides.
        :param X: Dataset
        :param y:
        :return: The output should be an array of weights
        """
        # endregion Summary

        # region Body

        # region Variant N

        kf = KFold(n_splits=self.k, shuffle=True)
        self.weights = []
        for model in self.estimators:
            weight = 0
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                weight += sum(predictions == y_test) / len(y_test)
            self.weights.append(weight / self.k)
        self.weights = np.array(self.weights) / sum(self.weights)

        # endregion Variant N

        # region Variant A

        # weights = []
        # for i in self.estimators:
        #     weights.append(np.mean(cross_val_score(i, X, y, scoring="accuracy")))
        # weights = np.array(weights)
        # weights = weights / weights.sum()
        # return weights

        # endregion Variant A

        # endregion Body

    def fit(self, X, y):
        # region Summary
        """
        Train the individual models on the whole training dataset and update "self.estimators" accordingly in order to
        use them for prediction
        :param X: Training dataset
        :param y:
        """
        # endregion Summary

        # region Body

        # region Variant N

        self.get_weights(X, y)
        for i, model in enumerate(self.estimators):
            model.fit(X, y)
            # self.estimators[i] = model

        # endregion Variant N

        # region Variant A

        # self.weights = self.get_weights(X, y)
        # for estimator in self.estimators:
        #     estimator.fit(X, y)

        # endregion Variant A

        # endregion Body

    def predict(self, X):
        # region Summary
        """
        Use the fitted individual models and their weights to perform prediction.
        This link may be useful: https://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting
        :param X: Dataset
        :return: Prediction for new instance
        """
        # endregion Summary

        # region Body

        # region Variant P

        pred = np.zeros((2, 180, 2))
        for i in range(self.nr_estimators):
            pred[i] = self.estimators[i].predict_proba(X) * self.weights[i]

        y_pred = np.argmax(np.mean(pred, axis=0), axis=1)
        return y_pred

        # endregion Variant P

        # region Variant N

        # probabilities = []
        # for i, model in enumerate(self.estimators):
        #     probabilities.append(self.weights[i] * model.predict_proba(X))
        #
        # probabilities = np.array(probabilities)
        # print(probabilities.shape)

        # endregion Variant N

        # region Variant A

        # y_predictions = self.estimators[0].predict_proba(X) * self.weights[0]
        # for i in range(1, len(self.estimators)):
        #     y_predictions += self.estimators[1].predict_proba(X) * self.weights[1]
        # y_predictions = y_predictions / len(self.weights)
        # y_predictions = np.argmax(y_predictions, axis=1)
        # return y_predictions

        # endregion Variant A

        # endregion Body

    # endregion Functions


class Stacking:
    # region Constructor

    def __init__(self, estimators, final_estimator, meta_features='class', cv=False, k=None):
        # region Summary
        """
        Constructor of Stacking class.
        :param estimators: List of classifier objects
        :param final_estimator: Classifier for the meta-model

        :param meta_features: Meta-features (input) of the meta-model. This should take the values:
        1. 'prob', if we take the class probabilities from the individual models. In this case, you need to use
        predict_proba() method on sklearn classifiers and discard one of the probability values. For example, if the
        task is a 2 class classification problem, then each individual model's predict_proba() method will return a
        vector of 2 values for each class's probability, and we can discard one of those values because it is the
        complement of the other class. ([p, q], where q = 1-p).
        In case we have an m-class classification problem and T individual models, then the input for the meta-model
        will be T * (m-1) dimensional vector, since each model will give m-1 probability values.
        2. 'class' if we take the predicted labels. In this case, the input for the meta-model will be a T dimensional vector.

        :param cv: Boolean specifying whether to use Cross Validation for deriving the meta-features or not, as described in the lecture slides
        :param k: Count of folds of Cross-Validation
        """
        # endregion Summary

        self.estimators = estimators
        self.nr_estimators = len(estimators)
        self.final_estimator = final_estimator
        self.meta_features = meta_features
        self.meta_X = None
        self.cv = cv
        self.k = k
        self.progressbar = progressbar.ProgressBar(widgets=widgets)

    # endregion Constructor

    # region Functions

    def fit(self, X, y):
        # region Summary
        """
        Derive the meta-features and train the meta-model on it
        :param X: Dataset
        :param y:
        """
        # endregion Summary

        # region Body

        # region Variant N

        self.nr_labels = len(np.unique(y))

        if self.meta_features == "class":
            meta_X = np.zeros((X.shape[0], self.nr_estimators))
        else:
            meta_X = np.zeros((X.shape[0], self.nr_estimators * (self.nr_labels - 1)))

        if self.cv:
            kf = KFold(n_splits=self.k, shuffle=True)

        for i in self.progressbar(range(self.nr_estimators)):
            if self.cv:
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    model = self.estimators[i]
                    model.fit(X_train, y_train)
                    if self.meta_features == "class":
                        meta_X[test_index, i] = model.predict(X_test)
                    else:
                        meta_X[test_index, i * (self.nr_labels - 1): (i + 1) * (self.nr_labels - 1)] = model.predict_proba(X_test)[:, 1:]
            else:
                model = self.estimators[i]
                model.fit(X, y)
                if self.meta_features == "class":
                    meta_X[:, i] = model.predict(X)
                else:
                    meta_X[:, i * (self.nr_labels - 1): (i + 1) * (self.nr_labels - 1)] = model.predict_proba(X_test)[:, 1:]

        self.meta_X = meta_X
        self.final_estimator.fit(meta_X, y)

        # endregion Variant N

        # region Variant A

        # X_meta = []
        # for estimator in self.estimators:
        #     estimator.fit(X, y)
        #     if self.meta_features == "class":
        #         if self.cv:
        #             X_meta.append(cross_val_predict(estimator, X, y, method="predict", cv=self.k))
        #         else:
        #             X_meta.append(estimator.predict(X))
        #     else:
        #         if self.cv:
        #             X_meta.append(cross_val_predict(estimator, X, y, method="predict_proba", cv=self.k))
        #         else:
        #             X_meta.append(estimator.predict_proba(X))
        # X_meta = np.array(X_meta)
        # if len(X_meta.shape) == 2:
        #     X_meta = X_meta.T
        # else:
        #     X_meta = X_meta.transpose([1, 0, 2]).reshape((len(X), -1))
        # self.final_estimator.fit(X_meta, y)

        # endregion Variant A

        # endregion Body

    def predict(self, X):
        # region Summary
        """
        Get the predictions of the individual models and provide them as inputs to the meta-model
        :param X: Dataset
        :return: Predictions of the individual models
        """
        # endregion Summary

        # region Body

        # region Variant N

        if self.meta_features == "class":
            meta_X = np.zeros((X.shape[0], self.nr_estimators))
        else:
            meta_X = np.zeros((X.shape[0], self.nr_estimators * (self.nr_labels - 1)))

        for i in range(self.nr_estimators):
            model = self.estimators[i]
            if self.meta_features == "class":
                meta_X[:, i] = model.predict(X)
            else:
                meta_X[:, i * (self.nr_labels - 1): (i + 1) * (self.nr_labels - 1)] = model.predict_proba(X)[:, 1:]

        return self.final_estimator.predict(meta_X)

        # endregion Variant N

        # region Variant A

        # X_meta = []
        # for estimator in self.estimators:
        #     if self.meta_features == 'class':
        #         X_meta.append(estimator.predict(X))
        #     else:
        #         X_meta.append(estimator.predict_proba(X))
        # X_meta = np.array(X_meta)
        # if len(X_meta.shape) == 2:
        #     X_meta = X_meta.T
        # else:
        #     X_meta = X_meta.transpose([1, 0, 2]).reshape((len(X), -1))
        # return self.final_estimator.predict(X_meta)

        # endregion Variant A

        # endregion Body

    # endregion Functions
