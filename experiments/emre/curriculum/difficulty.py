"""How hard is a dataset?

Two ways to answer that, and we use both for different jobs:

  1. difficulty_score(...) looks only at the prior *knobs* (features, classes,
     noise). It's cheap and we use it to plan the curriculum stages on paper.

  2. measured_difficulty(X, y) looks at the data that actually came out. It's
     what we plot to check that "easy" really does look easier.

Both return a number in [0, 1] where higher means harder.
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def learnability_difficulty(X, y, k=15, test_frac=0.4, seed=0):
    """Difficulty = how badly a cheap model does on the task. Higher = harder.

    We fit a small kNN on part of the data and score balanced accuracy on a
    held-out part; difficulty is 1 - that accuracy. This asks the question that
    actually matters - "is this function learnable?" - instead of the geometric
    "are the class blobs separated?", which turned out to saturate on these SCM
    datasets (every knob looked equally hard). 0.5 = coin-flip on binary, 0 = solved.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return 0.0

    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - X.mean(axis=0)) / std

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xn, y, test_size=test_frac, random_state=seed, stratify=y
        )
    except ValueError:
        # some class had too few samples to stratify - treat as degenerate/hard
        return 1.0

    clf = KNeighborsClassifier(n_neighbors=min(k, len(X_tr))).fit(X_tr, y_tr)
    acc = balanced_accuracy_score(y_te, clf.predict(X_te))
    return float(np.clip(1.0 - acc, 0.0, 1.0))


def difficulty_score(num_features, num_classes, noise_std=0.0,
                     max_features=100, max_classes=10, max_noise=0.3):
    """Difficulty straight from the prior knobs.

    Just a weighted sum: more features, more classes and more noise all make
    things harder. The weights (40/30/30) are a starting point — we can tune
    them once we see how the stages behave.
    """
    f = (num_features / max_features) * 0.4
    c = (num_classes / max_classes) * 0.3
    n = (noise_std / max_noise) * 0.3
    return float(min(f + c + n, 1.0))


def class_separation(X, y):
    """How well separated are the classes? Bigger = easier.

    We take the distance between class centroids and divide it by how spread out
    the points are *within* each class. That ratio (it's basically a Fisher
    score) is the part that matters: classes far apart relative to their own
    scatter are easy to tell apart.

    Dividing by the within-class spread also cancels a trap we hit at first - raw
    centroid distance grows just because there are more features, which made
    high-dimensional ("hard") datasets look deceptively easy.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    std = X.std(axis=0)
    std[std == 0] = 1.0  # leave constant columns alone instead of dividing by 0
    Xn = (X - X.mean(axis=0)) / std

    classes = np.unique(y)
    if len(classes) < 2:
        return 0.0

    centroids = np.stack([Xn[y == c].mean(axis=0) for c in classes])

    # how far each point sits from its own class centroid, averaged over classes
    within = np.mean([
        np.linalg.norm(Xn[y == c] - centroids[i], axis=1).mean()
        for i, c in enumerate(classes)
    ])

    # mean pairwise distance between the centroids
    between = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            between.append(np.linalg.norm(centroids[i] - centroids[j]))
    between = np.mean(between)

    return float(between / (within + 1e-8))


def measured_difficulty(X, y):
    """Difficulty read off the generated data.

    We turn class separation into a 0..1 score: well separated classes -> near 0
    (easy), overlapping classes -> near 1 (hard).
    """
    sep = class_separation(X, y)
    return float(1.0 / (1.0 + sep))
