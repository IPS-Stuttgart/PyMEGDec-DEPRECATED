import numpy as np
import scipy.io as sio
from pymegdec.classifiers import (
    get_default_classifier_param,
    should_use_default_classifier_param,
    train_multiclass_classifier,
)
from pymegdec.data_config import resolve_data_folder
from pymegdec.preprocessing import preprocess_features
from neureptrace.decoding.transfer import cross_validate_feature_decoding


# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def cross_validate_single_dataset(
    data_folder,
    participant_id,
    n_folds=10,
    window_size=0.1,
    train_window_center=0.2,
    null_window_center=-0.2,
    new_framerate=float("inf"),
    classifier="multiclass-svm",
    classifier_param=np.nan,
    components_pca=100,
    frequency_range=(0, float("inf")),
    random_state=None,
):

    if should_use_default_classifier_param(classifier_param):
        classifier_param = get_default_classifier_param(classifier)

    data_folder = resolve_data_folder(data_folder)

    data = sio.loadmat(f"{data_folder}/Part{participant_id}Data.mat")["data"][0]
    labels = np.asarray(data["trialinfo"][0][0], dtype=int).ravel()
    if np.isnan(null_window_center):
        # Match evaluate_model_transfer/stimulus decoding: without null
        # features there is no additional null class, so class labels should
        # be zero-based for classifiers that require contiguous labels.
        labels = labels - 1
    stimuli_features, null_features = preprocess_features(
        data,
        frequency_range,
        new_framerate,
        window_size,
        train_window_center,
        null_window_center,
    )
    stimulus_feature_matrix = np.hstack(stimuli_features).T
    null_feature_matrix = np.hstack(null_features).T if null_features else None
    result = cross_validate_feature_decoding(
        stimulus_feature_matrix,
        labels,
        null_features=null_feature_matrix,
        n_folds=n_folds,
        classifier=classifier,
        classifier_param=classifier_param,
        components_pca=components_pca,
        random_state=random_state,
        fit_model=lambda features, train_labels: train_multiclass_classifier(
            features,
            train_labels,
            classifier,
            classifier_param,
            random_state=random_state,
        ),
    )
    accuracy = result.accuracy
    print(f"Participant {participant_id}: {accuracy * 100:.2f}% accuracy")

    return accuracy


# pylint: enable=too-many-arguments,too-many-positional-arguments
# pylint: enable=too-many-locals,too-many-branches,too-many-statements


if __name__ == "__main__":
    acc = cross_validate_single_dataset(
        r".",
        2,
        classifier="multiclass-svm",
        components_pca=100,
    )
    print(acc)
