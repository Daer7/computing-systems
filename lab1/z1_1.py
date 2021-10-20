import numpy as np
from scipy.sparse import data
from sklearn.model_selection import RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score
import threading


class DataSetSolver(threading.Thread):

    clfs = {'GNB': GaussianNB(), 'kNN': KNeighborsClassifier()}
    n_splits = 5
    n_repeats = 2
    rskf = RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scores = np.zeros((len(clfs), 30, n_splits * n_repeats))

    def __init__(self, data_id, dataset):
        threading.Thread.__init__(self)
        self.data_id = data_id
        self.dataset = dataset

    def run(self):
        dataset = np.genfromtxt(
            "datasets/{}".format(self.dataset), delimiter=',')
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)

        for fold_id, (train, test) in enumerate(DataSetSolver.rskf.split(X, y)):
            for clf_id, clf_name in enumerate(DataSetSolver.clfs):
                clf = clone(DataSetSolver.clfs[clf_name])
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                DataSetSolver.scores[clf_id, self.data_id, fold_id] = accuracy_score(
                    y[test], y_pred)


if __name__ == "__main__":
    datasets = ['appendicitis.csv', 'australian.csv', 'balance.csv',
                'banknote.csv', 'breastcan.csv', 'breastcancoimbra.csv', 'bupa.csv', 'coil2000.csv', 'cryotherapy.csv', 'ecoli4.csv',
                'german.csv', 'glass2.csv', 'glass4.csv', 'glass5.csv', 'haberman.csv', 'heart.csv', 'ionosphere.csv', 'iris.csv', 'liver.csv',
                'mammographic.csv', 'monk-2.csv', 'phoneme.csv', 'pima.csv', 'popfailures.csv', 'ring.csv', 'sonar.csv', 'soybean.csv',
                'spambase.csv', 'titanic.csv', 'wine.csv']

    threads = []
    for data_id, dataset in enumerate(datasets):
        threads.append(DataSetSolver(data_id, dataset))
        threads[-1].start()

    for t in threads:
        t.join()

    # usrednienie po foldach i transpozycja
    mean_scores = np.mean(DataSetSolver.scores, axis=2).T
    np.save('results_1_1', mean_scores)
