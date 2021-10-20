import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score

clfs = {'GNB': GaussianNB(), 'kNN': KNeighborsClassifier()}

datasets = ['appendicitis.csv', 'australian.csv', 'balance.csv',
            'banknote.csv', 'breastcan.csv', 'breastcancoimbra.csv', 'bupa.csv', 'coil2000.csv', 'cryotherapy.csv', 'ecoli4.csv',
            'german.csv', 'glass2.csv', 'glass4.csv', 'glass5.csv', 'haberman.csv', 'heart.csv', 'ionosphere.csv', 'iris.csv', 'liver.csv',
            'mammographic.csv', 'monk-2.csv', 'phoneme.csv', 'pima.csv', 'popfailures.csv', 'ring.csv', 'sonar.csv', 'soybean.csv',
            'spambase.csv', 'titanic.csv', 'wine.csv']
n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)

scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/{}".format(dataset), delimiter=',')
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

# usrednienie po foldach i transpozycja
mean_scores = np.mean(scores, axis=2).T
np.save('results_1_0', mean_scores)
