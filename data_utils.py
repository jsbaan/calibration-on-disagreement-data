from typing import List, Optional, Any

import pandas as pd
import numpy as np
import json
import collections

from scipy.special import softmax


def load_chaosnli(sources: List[str]):
    """
    Loads chaosNLI annotations for a given source dataset.
    Args:
        sources: ["snli", "mnli", "alphanli"]

    Returns: a list json objects that contain annotations and information about each instance.
    """

    def _load(_source):
        with open(f"data/chaosNLI_v1.0/chaosNLI_{_source}.jsonl", "rb") as f:
            _data = f.readlines()
            return [json.loads(datum) for datum in _data]

    data = list()
    for source in sources:
        data += _load(source)
    return data


def load_predictions(
    model_name: str,
    model_seed: int,
    datasets: List[str],
    splits: List[str],
):
    """
    Loads model predictions for a given model, seed, dataset and split.
    Args:
        model_name: ["roberta-base", "bert-base-uncased"]
        model_seed: [0, 1, 2]
        datasets: ["snli", "mnli"]
        splits: ["dev", "test"]

    Returns: a list of dictionaries containing predictions information for each instance.

    """

    def _load(_dataset, _split):
        with open(
            f"data/predictions/{_dataset.upper()}_{_split}_{model_name}_{model_seed}.json",
            "r",
        ) as f:
            data = f.readlines()
            return [json.loads(datum) for datum in data]

    predictions = list()
    for dataset in datasets:
        for split in splits:
            predictions += _load(dataset, split)
    return predictions


def load_data(
    datasets: List[str],
    splits: List[str],
    model_name: str,
    model_seed: int,
    temp: int = 1,
    np_rng: Optional[np.random.RandomState] = None,
):
    """
    Loads chaosNLI annotations and predictions for a given model. Aligns the two both class-wise and instance-wise.
    Args:
        datasets: ["snli", "mnli"]
        splits: ["dev", "test"]
        model_name: ["bert-base-uncased", "roberta-base"]
        model_seed: [0, 1, 2]
        temp: temperature to scale logits with
        np_rng: a numpy random number generator

    Returns:
        annotations_dict: Contains the original 5 annotations and additional 100 annotations for each instance
            [n_instances, n_classes]
        cls_probs: Contains predicted probabilities for each instance [n_instances, n_classes]

    """
    # Load predictions for the test and/or dev splits of SNLI and/or MNLI
    pred = load_predictions(model_name, model_seed, datasets, splits)
    df_pred = pd.DataFrame.from_records(pred, index=["uid"])

    # Load ChaosNLI instances originating from SNLI and/or MNLI
    chaosnli = load_chaosnli(datasets)
    df_chaosnli = pd.DataFrame.from_records(chaosnli, index=["uid"])

    # Keep only predictions for instances that are in the chaosNLI dataset
    # Note that pandas' merge sorts the resulting df based on the key, which is uid in this case
    df_pred = df_pred.merge(df_chaosnli, on="uid", how="inner")
    if np_rng is not None:
        df_pred.sample(frac=1, random_state=np_rng).reset_index(drop=True)

    # The class order differs between predictions and chaosNLI; we align them here
    df_pred["logits"] = df_pred.logits.apply(lambda x: [x[0], x[2], x[1]])
    df_pred["probs"] = df_pred.probs.apply(lambda x: [x[0], x[2], x[1]])
    df_pred["true"] = df_pred.true.apply(lambda x: {0: 0, 1: 2, 2: 1}[x])
    df_pred["pred"] = df_pred.pred.apply(lambda x: {0: 0, 1: 2, 2: 1}[x])

    # Extract predictions
    cls_logits = np.array(df_pred.logits.values.tolist())
    cls_probs = softmax(cls_logits / temp, axis=1)

    # Extract annotations
    chaosnli_annotations = np.array(df_pred.label_count.values.tolist())
    original_annotations = [collections.Counter(x) for x in df_pred.old_labels.values.tolist()]
    original_annotations = np.array(
        [[c["entailment"], c["neutral"], c["contradiction"]] for c in original_annotations]
    )

    annotations_dict = {
        "original": original_annotations,
        "chaosnli": chaosnli_annotations,
    }
    return annotations_dict, cls_probs


def subsample_annotations(annotations: np.ndarray, subsample_size: int, np_rng):
    """
    Subsample human votes for each instance given the 100 annotations for each instance provided by chaosNLI.
    Args:
        annotations: [num_instances, num_classes]
        subsample_size: the number of votes to subsample from each instance
        np_rng: a numpy random number generator

    Returns: Each subsample_idx contains `subsample_size` votes per instance.
        [subsample_idx, num_instances, num_classes]

    """
    N, C = annotations.shape
    T = annotations.sum(-1)[0]
    n_subsamples = int(T / subsample_size)
    subsampled_annotations = []
    for i in range(N):
        # Flatten count vectors into 100 individual votes
        flattened_instance_counts = []
        for c in range(C):
            flattened_instance_counts = flattened_instance_counts + (annotations[i, c] * [c])
        flattened_instance_counts = np.array(flattened_instance_counts)

        # Shuffle the votes and partition them into `n_subsamples` samples of length `subsample_size`
        shuffled_indices = np.arange(T)
        np_rng.shuffle(shuffled_indices)
        samples = [
            flattened_instance_counts[shuffled_indices[j : j + subsample_size]]
            for j in range(0, T, subsample_size)
        ]

        assert np.all(np.array([len(s) for s in samples]) == subsample_size)
        assert len(samples) == n_subsamples

        # Construct count vectors given the samples of individual votes
        instance_subsamples = []
        for s in range(n_subsamples):
            counter = collections.Counter(samples[s])
            subsample = np.zeros(3)
            for c in range(C):
                subsample[c] = counter.get(c, 0)
            instance_subsamples.append(subsample)
        subsampled_annotations.append(instance_subsamples)
    subsampled_annotations = np.array(subsampled_annotations).astype(int)

    # The counts for each class for each instance should be the same
    np.testing.assert_array_equal(annotations, subsampled_annotations.sum(1))
    return subsampled_annotations.swapaxes(0, 1)
