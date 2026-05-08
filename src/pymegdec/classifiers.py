"""Classifier compatibility wrappers used by PyMEGDec decoding routines."""

import numpy as np
from reptrace.decoding.classifiers import (
    CLASSIFIER_REGISTRY as REPTRACE_CLASSIFIER_REGISTRY,
)
from reptrace.decoding.classifiers import (
    DEFAULT_CLASSIFIER_PARAMS as REPTRACE_DEFAULT_CLASSIFIER_PARAMS,
)
from reptrace.decoding.classifiers import (
    ClassifierSpec,
)
from reptrace.decoding.classifiers import (
    get_default_classifier_param as get_reptrace_default_classifier_param,
)
from reptrace.decoding.classifiers import (
    should_use_default_classifier_param,
    train_binary_svm,
)
from reptrace.decoding.classifiers import train_classifier as train_reptrace_classifier
from reptrace.decoding.classifiers import (
    train_gradient_boosting,
    train_lasso_logistic,
)

_PYMEGDEC_DEFAULT_CLASSIFIER_PARAMS = {
    "xgboost": 100,
    "pytorch-mlp": {
        "hidden_dim": 720,
        "max_epochs": 500,
        "learning_rate": 1e-3,
        "dropout_rate": 0.2,
        "random_seed": 0,
    },
}
_DEFAULT_CLASSIFIER_PARAMS = {
    **REPTRACE_DEFAULT_CLASSIFIER_PARAMS,
    **_PYMEGDEC_DEFAULT_CLASSIFIER_PARAMS,
}
__all__ = [
    "CLASSIFIER_REGISTRY",
    "ClassifierSpec",
    "get_default_classifier_param",
    "should_use_default_classifier_param",
    "train_binary_svm",
    "train_for_stimulus_lasso_glm",
    "train_gradient_boosting",
    "train_multiclass_classifier",
]


def __getattr__(name):
    if name == "MLPClassifierTorch":
        from pymegdec.torch_models import MLPClassifierTorch

        return MLPClassifierTorch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _build_xgboost(_features, _labels, classifier_param, random_state):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("Install PyMEGDec with the xgboost extra to use classifier='xgboost'.") from exc

    return xgb.XGBClassifier(
        n_estimators=int(classifier_param),
        eval_metric="mlogloss",
        random_state=random_state,
    )


def _build_pytorch_mlp_classifier(features, labels, classifier_param, random_state):
    return _train_pytorch_mlp(
        features,
        labels,
        classifier_param,
        random_state=random_state,
    )


CLASSIFIER_REGISTRY = {
    **REPTRACE_CLASSIFIER_REGISTRY,
    "xgboost": ClassifierSpec(_build_xgboost),
    "pytorch-mlp": ClassifierSpec(_build_pytorch_mlp_classifier, fits_in_builder=True),
}


def train_multiclass_classifier(
    features,
    labels,
    classifier,
    classifier_param,
    random_state=None,
):
    return train_reptrace_classifier(
        features,
        labels,
        classifier,
        classifier_param,
        random_state=random_state,
        registry=CLASSIFIER_REGISTRY,
    )


def _train_pytorch_mlp(features, labels, classifier_param, random_state=None):
    random_seed = _resolve_pytorch_random_seed(classifier_param, random_state)
    if random_seed is not None:
        _seed_pytorch_training(random_seed)

    model = _build_pytorch_mlp(features, labels, classifier_param)
    train_loader, val_loader = _build_pytorch_data_loaders(features, labels, random_seed=random_seed)
    trainer = _build_pytorch_trainer(classifier_param, random_seed=random_seed)
    trainer.fit(model, train_loader, val_loader)
    return model


def _resolve_pytorch_random_seed(classifier_param, random_state):
    random_seed = random_state
    if random_seed is None:
        random_seed = classifier_param.get("random_seed")
    if random_seed is None:
        return None
    return int(random_seed)


def _seed_pytorch_training(random_seed):
    try:
        import pytorch_lightning as pl
    except ImportError as exc:
        raise ImportError("Install PyMEGDec with the torch extra to use classifier='pytorch-mlp'.") from exc

    pl.seed_everything(random_seed, workers=True)


def _build_pytorch_mlp(features, labels, classifier_param):
    try:
        from pymegdec.torch_models import MLPClassifierTorch
    except ImportError as exc:
        raise ImportError("Install PyMEGDec with the torch extra to use classifier='pytorch-mlp'.") from exc

    return MLPClassifierTorch(
        features.shape[1],
        int(classifier_param["hidden_dim"]),
        len(np.unique(labels)),
        learning_rate=classifier_param["learning_rate"],
        dropout_rate=classifier_param["dropout_rate"],
    )


def _build_pytorch_data_loaders(features, labels, *, random_seed=None):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install PyMEGDec with the torch extra to use classifier='pytorch-mlp'.") from exc

    train_dataset, val_dataset = _split_pytorch_dataset(torch, features, labels, random_seed)
    train_generator = _build_torch_generator(torch, random_seed)
    return (
        torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, generator=train_generator),
        torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False),
    )


def _split_pytorch_dataset(torch, features, labels, random_seed):
    full_dataset = torch.utils.data.TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    split_generator = _build_torch_generator(torch, random_seed)
    return torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=split_generator)


def _build_torch_generator(torch, random_seed):
    if random_seed is None:
        return None

    generator = torch.Generator()
    generator.manual_seed(int(random_seed))
    return generator


def _build_pytorch_trainer(classifier_param, *, random_seed=None):
    try:
        import pytorch_lightning as pl
    except ImportError as exc:
        raise ImportError("Install PyMEGDec with the torch extra to use classifier='pytorch-mlp'.") from exc

    return pl.Trainer(
        max_epochs=int(classifier_param["max_epochs"]),
        default_root_dir=r"lightning_logs",
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=10)],
        deterministic=random_seed is not None,
    )


def train_for_stimulus_lasso_glm(
    train_features,
    train_labels,
    lambda_,
    random_state=None,
):
    return train_lasso_logistic(
        train_features,
        train_labels,
        lambda_,
        random_state=random_state,
    )


def get_default_classifier_param(classifier):
    if classifier in _PYMEGDEC_DEFAULT_CLASSIFIER_PARAMS:
        classifier_param = _PYMEGDEC_DEFAULT_CLASSIFIER_PARAMS[classifier]
        if isinstance(classifier_param, dict):
            return classifier_param.copy()
        return classifier_param
    return get_reptrace_default_classifier_param(classifier)
