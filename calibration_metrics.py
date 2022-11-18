from typing import Optional

import numpy as np
from scipy.stats import entropy
from sklearn.metrics import f1_score


def tvd(model_probs: np.ndarray, human_probs: np.ndarray, mean_per: Optional[str] = None):
    """
    Computes TVD scores allowing for multiple sub-samples and groups (=classifiers).

    p: classifiers [G, 1, N, C]
    q: MLE given (sub-samples of) annotations [1, S, N, C]

    returns:
        tvd: [G, S, N] (mean_per=None), [G, S] (mean_per=sample), [G, N] (mean_per=instance)
    """
    assert model_probs.max() <= 1.0 and model_probs.min() >= 0
    assert human_probs.max() <= 1.0 and human_probs.min() >= 0

    tvds = np.sum(np.abs(model_probs - human_probs), axis=-1) / 2
    if mean_per is not None:
        if mean_per == "instance":
            tvds = tvds.mean(1)
        elif mean_per == "sample":
            tvds = tvds.mean(2)
    return tvds


def acck(model_probs: np.ndarray, human_votes: np.ndarray, mean_per: str, k: int = 1):
    """
    Computes accuracy-k scores allowing for multiple sub-samples and multiple groups (=classifiers).

    model_probs: classifiers [G, 1, N, C]
    human_votes: annotations [1, S, N, C]
    returns:
        accks_scores: [G, S] (mean_per=sample ) or [G, N] (mean_per=instance)

    """
    # [G, S, N]
    pred = np.argsort(model_probs, -1)[..., -k]
    # [G, S, N]
    gold = np.argsort(human_votes, -1)[..., -k]
    # [G, S, N]
    comp = pred == gold
    return np.array(comp.mean(2 if mean_per == "sample" else 1))


def f1_samples(
    model_probs: np.ndarray,
    human_votes: np.ndarray,
    n_samples: int = None,
    average: str = "macro",
):
    """
    Compute f1 scores allowing for multiple sub-samples and multiple groups (=classifiers).

    model_probs: [G, N, C]
    human_votes: [S, N, C]
    n_samples: the number of samples used to compute the f1 score on, for computational reasons.
    returns:
        f1_scores: [G, S]
    """
    G = model_probs.shape[0]
    S = human_votes.shape[0] if not n_samples else n_samples

    f1_scores = []
    for c in range(G):
        sample_stats = []
        for s in range(S):
            sample_stats.append(
                f1_score(
                    np.argmax(human_votes[s, ...], -1),
                    np.argmax(model_probs[c, ...], -1),
                    average=average,
                )
            )
        f1_scores.append(sample_stats)
    return np.array(f1_scores)


def ece(probs, labels, n_bins=10, k=None):
    """
    Computes classwise or confidence ECE for one sub-sample of annotations and one group (=classifier).

    probs: classifier probs [N, C]
    labels: argmax of annotations [N]
    n_bins: the number of bins
    k: the class {0, ..., k-1} to measure calibration for. None for confidence calibration

    returns:
        ece: [1]
    """

    bucket_stats = []
    if k is None:
        confs = np.max(probs, -1)
        preds = np.argmax(probs, -1)
    else:
        confs = probs[:, k]
        preds = k * np.ones(labels.shape)

    # iterate over bins with confidence range (a, b)
    for i, b in enumerate(np.linspace(0.0, 1.0, n_bins + 1)):
        if i == 0:
            a = b
            continue
        idxs = np.where((confs > a) & (confs <= b))[0]
        if idxs.shape != (0,):
            bucket_conf = np.mean(confs[idxs])
            bucket_acc = np.mean(labels[idxs] == preds[idxs])
            bucket_stats.append([bucket_conf, bucket_acc, len(idxs)])
        a = b
    s = np.array(bucket_stats)
    return np.average(np.abs(s[:, 0] - s[:, 1]), weights=s[:, 2])


def ece_samples(model_probs: np.ndarray, human_votes: np.ndarray, n_samples: int = None):
    """
    Computes ece for multiple sub-samples and multiple groups (=classifiers).

    model_probs: [G, N, C]
    human_votes: [S, N, C]
    n_samples: to spare compute, use less samples than available

    returns:
        ece_scores: [G, S]
    """
    G = model_probs.shape[0]
    S = human_votes.shape[0] if not n_samples else n_samples
    ece_scores = []
    for c in range(G):
        sample_stats = []
        for s in range(S):
            sample_stats.append(ece(model_probs[c, ...], np.argmax(human_votes[s, ...], -1)))
        ece_scores.append(sample_stats)
    return np.array(ece_scores)


def ent_ce(model_probs: np.ndarray, human_probs: np.ndarray, absolute: bool = True):
    """
    Compute entropy calibration error for multiple groups (=classifiers).

    Args:
        model_probs: [G, N, C]
        human_probs: [N, C]
        absolute: whether to take the absolute value of the error

    Returns:
        ent_ce: [G]
    """
    ent_ces = list()
    for cls in range(model_probs.shape[0]):
        ent_ces_cls = list()
        for i in range(model_probs.shape[1]):
            pred_dist = model_probs[cls, i, :]
            true_dist = human_probs[i, :]
            error = entropy(pred_dist) - entropy(true_dist)
            ent_ces_cls.append(abs(error) if absolute else error)
        ent_ces.append(ent_ces_cls)
    return np.array(ent_ces)
