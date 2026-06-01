import warnings

import numpy as np
import scipy.io as sio
from pymegdec.classifiers import (
    get_default_classifier_param,
    should_use_default_classifier_param,
    train_multiclass_classifier,
)
from pymegdec.data_config import resolve_data_folder
from pymegdec.preprocessing import preprocess_features
from neureptrace.decoding.transfer import evaluate_feature_transfer


def _trialinfo_labels(data):
    return np.array(data["trialinfo"][0][0], dtype=int, copy=True).ravel()


# jscpd:ignore-start
# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def evaluate_model_transfer(
    data_folder,
    parts,
    window_size=0.1,
    train_window_center=0.2,
    null_window_center=-0.2,
    new_framerate=float("inf"),
    classifier="multiclass-svm",
    classifier_param=np.nan,
    components_pca=100,
    frequency_range=(0, float("inf")),
    random_state=None,
    return_feature_importance=False,
):
    # jscpd:ignore-end

    if should_use_default_classifier_param(classifier_param):
        classifier_param = get_default_classifier_param(classifier)

    data_folder = resolve_data_folder(data_folder)

    train_exp_data = sio.loadmat(f"{data_folder}/Part{parts}Data.mat")["data"][0]
    val_exp_data = sio.loadmat(f"{data_folder}/Part{parts}CueData.mat")["data"][0]

    labels_train_exp = _trialinfo_labels(train_exp_data)
    labels_val_exp = _trialinfo_labels(val_exp_data)
    if np.isnan(null_window_center):
        # There is no null data in the validation experiment, and some
        # classifiers do not support labels starting above 0.
        labels_train_exp = labels_train_exp - 1
        labels_val_exp = labels_val_exp - 1

    train_sample_interval = np.diff(train_exp_data["time"][0][0][0][0, :2])
    val_sample_interval = np.diff(val_exp_data["time"][0][0][0][0, :2])
    if not np.allclose(train_sample_interval, val_sample_interval):
        raise ValueError("Sampling rate of the two experiments must match.")

    if not np.array_equal(np.unique(labels_train_exp), np.unique(labels_val_exp)):
        warnings.warn("There are labels in the training or validation experiment " "that are not in the other experiment.")

    stimuli_features_train_exp, null_features_train_exp = preprocess_features(
        train_exp_data,
        frequency_range,
        new_framerate,
        window_size,
        train_window_center,
        null_window_center,
    )
    stimuli_features_val_exp, _ = preprocess_features(
        val_exp_data,
        frequency_range,
        new_framerate,
        window_size,
        train_window_center,
        np.nan,
    )

    features_train_exp = np.hstack(stimuli_features_train_exp).T
    train_null_feature_matrix = np.hstack(null_features_train_exp).T if null_features_train_exp else None
    features_val_exp = np.hstack(stimuli_features_val_exp).T

    result = evaluate_feature_transfer(
        features_train_exp,
        labels_train_exp,
        features_val_exp,
        labels_val_exp,
        train_null_features=train_null_feature_matrix,
        classifier=classifier,
        classifier_param=classifier_param,
        components_pca=components_pca,
        random_state=random_state,
        fit_model=lambda features, labels: train_multiclass_classifier(
            features,
            labels,
            classifier,
            classifier_param,
            random_state=random_state,
        ),
    )
    if components_pca != float("inf"):
        print("Explained Variance by " f"{components_pca} components: {result.model_bundle.explained_variance_percent:.2f}%")

    accuracy = result.accuracy
    if return_feature_importance:
        return accuracy, get_original_feature_importance(result.model_bundle.model, result.model_bundle.pca_coeff)

    return accuracy


def get_original_feature_importance(model, pca_components=None):
    feature_importance = np.asarray(_get_classifier_coefficients(model))
    if feature_importance.ndim == 1:
        feature_importance = feature_importance.reshape(1, -1)
    if feature_importance.ndim != 2:
        raise ValueError("Feature importance coefficients must be a one- or two-dimensional array.")

    if pca_components is not None:
        feature_importance = _project_pca_feature_importance(feature_importance, pca_components)

    return feature_importance


def _project_pca_feature_importance(feature_importance, pca_components):
    """Map linear classifier coefficients from PCA coordinates to input coordinates.

    The transfer helper may expose PCA loadings in either the MATLAB convention
    ``(n_original_features, n_components)`` or the sklearn convention
    ``(n_components, n_original_features)``.  For a linear model fitted on PCA
    scores, the original-space coefficients are obtained by composing with the
    same forward projection matrix; this is not a pseudoinverse operation.
    """

    pca_components = np.asarray(pca_components)
    if pca_components.ndim != 2:
        raise ValueError("PCA components must be a two-dimensional array.")

    n_pca_features = feature_importance.shape[1]
    if pca_components.shape[0] == n_pca_features:
        # sklearn-style components_: scores = centered_features @ components_.T
        return feature_importance @ pca_components

    if pca_components.shape[1] == n_pca_features:
        # MATLAB-style coeff/loadings: scores = centered_features @ coeff
        return feature_importance @ pca_components.T

    raise ValueError(
        "PCA component shape "
        f"{pca_components.shape} is incompatible with classifier coefficients "
        f"of shape {feature_importance.shape}."
    )


def _get_classifier_coefficients(model):
    if hasattr(model, "coef_"):
        return model.coef_

    if hasattr(model, "steps"):
        classifier = model.steps[-1][1]
        if hasattr(classifier, "coef_"):
            coefficients = classifier.coef_
            for _, transformer in model.steps[:-1]:
                if hasattr(transformer, "scale_"):
                    coefficients = coefficients / transformer.scale_
            return coefficients

    raise ValueError("Feature importance is only available for linear classifiers with coefficients.")


if __name__ == "__main__":
    acc = evaluate_model_transfer(
        r".",
        2,
        classifier="multiclass-svm",
        components_pca=100,
    )
    print(acc)
