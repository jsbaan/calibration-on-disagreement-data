from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import entropy
from calibration_metrics import tvd, ece_samples, acck, ent_ce


def mle(counts: np.ndarray, eps: float = 0):
    """
    Maximum likelihood estimate of the parameters for a categorical probability distribution from counts.
    Args:
        counts: contains human votes per class for each instance. [n_instances, n_classes]
        eps: int

    Returns: A categorical distribution over classes [n_instances, n_classes]

    """
    counts = counts + eps
    return counts / np.sum(counts, -1, keepdims=True)


def divergence(hist_p: np.ndarray, hist_q: np.ndarray, div: str):
    """
    Takes in two histograms (array with the number of instances per bin) with the same bins and returns a divergence.
    Args:
        hist_p: [n_bins]
        hist_q: [n_bins]
        div: kl or tvd

    Returns: float divergence between probability distributions (normalized histograms).

    """
    # p and q are histograms: normalize first
    p = hist_p / hist_p.sum()
    q = hist_q / hist_q.sum()
    if div == "kl":
        return entropy(p, q)
    elif div == "tvd":
        return np.sum(np.abs(p - q), axis=-1) / 2


def generate_figure2_and_table2(
    classifiers: np.ndarray,
    classifier_names: List[str],
    annotations: np.ndarray,
    n_bins: int,
    div: str,
):
    """
    Generates Figure 2 and Table 2 in the paper.
    Args:
        classifiers: predicted probabilities [n_classifiers, n_instances, n_classes]
        classifier_names: names of classifiers
        annotations: human votes [n_instances, n_classes]
        n_bins: number of bins
        div: divergence to use
    """
    human_judgement_distribution = mle(annotations)
    tvds = tvd(classifiers[:, None, ...], human_judgement_distribution[None, None, ...]).squeeze(1)
    ece_score = ece_samples(classifiers, annotations[None, ...]).flatten()

    # Plot histograms
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    ax = axes
    barplot = sns.histplot(
        data={classifier_names[i]: tvds[i, :] for i in [0, 1, 2, 3]},
        ax=ax,
        bins=n_bins,
        kde=True,
        stat="probability",
    )
    ax.set(xlabel="DistCE (TVD)")
    ax.lines[2].set_linestyle("dashed")
    ax.lines[3].set_linestyle("dashed")

    for i, bar in enumerate(barplot.patches):
        if i >= 2 * n_bins:
            bar.set_alpha(0.3)
        else:
            bar.set_alpha(0.6)

    # Compute histograms
    tvd_hist_ideal_cls1, _ = np.histogram(tvds[0, :], bins=n_bins, range=(0, 1))
    tvd_hist_ideal_cls2, _ = np.histogram(tvds[1, :], bins=n_bins, range=(0, 1))
    tvd_hist_roberta, _ = np.histogram(tvds[2, :], bins=n_bins, range=(0, 1))
    tvd_hist_roberta_ts, _ = np.histogram(tvds[3, :], bins=n_bins, range=(0, 1))

    # Print results
    print(classifier_names)
    print("ECE scores", ece_score)
    print("DistCE scores", tvds.mean(axis=1))
    div1 = divergence(tvd_hist_ideal_cls1, tvd_hist_ideal_cls2, div=div)
    div2 = divergence(tvd_hist_ideal_cls1, tvd_hist_roberta, div=div)
    div3 = divergence(tvd_hist_ideal_cls1, tvd_hist_roberta_ts, div=div)
    print(f"{div}(H1, H2):{div1:.4f}")
    print(f"{div}(H1, vanilla): {div2:.2f}")
    print(f"{div}(H1, TS): {div3:.2f}")


def generate_table1(classifiers: np.ndarray, classifier_names: List[str], annotations: np.ndarray):
    """
    Generates Table 1 in the paper.
    Args:
        classifiers: predicted probabilities [n_classifiers, n_instances, n_classes]
        classifier_names: names of classifiers
        annotations: human votes [n_instances, n_classes]
    """
    human_judgement_distribution = mle(annotations)
    distces = (
        tvd(classifiers[:, None, ...], human_judgement_distribution[None, None, ...])
        .squeeze(1)
        .mean(axis=1)
    )
    eces = ece_samples(classifiers, annotations[None, ...]).flatten()
    accs = acck(
        classifiers[:, None, ...], annotations[None, None, ...], mean_per="sample"
    ).flatten()
    rankces = np.all(classifiers.argsort(-1) == annotations.argsort(-1), axis=-1).mean(-1)
    entces = ent_ce(classifiers, human_judgement_distribution, absolute=True).mean(-1)

    print(classifier_names)
    for name, metric in [
        ("Acc", accs),
        ("ECE", eces),
        ("RankCS", rankces),
        ("EntCE", entces),
        ("DistCE", distces),
    ]:
        print(name, metric)


def generate_figure1(
    classifier: np.ndarray, classifier_name: str, annotations: np.ndarray, n_bins: int
):
    """
    Generates Figure 1 in the paper.
    Args:
        classifier: predicted probabilities [n_instances, n_classes]
        classifier_name: name of classifier
        annotations: human votes [n_instances, n_classes]
        n_bins: number of bins
    """
    human_judgement_distribution = mle(annotations)
    dist_ce = tvd(classifier[None, None, ...], human_judgement_distribution[None, None, :])

    # Plot histogram
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    sns.histplot(
        dist_ce[0, 0, :],
        # ax=axes,
        binwidth=1 / n_bins,
        binrange=(0, 1),
        kde=True,
        stat="probability",
    )
    axes.set(xlabel="DistCE (TVD)")
    axes.set(title=classifier_name)
    axes.set(ylim=(0, 0.135))
