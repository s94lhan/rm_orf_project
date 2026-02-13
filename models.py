
import numpy as np
import warnings
import inspect
from typing import Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression

from config import (
    DEFAULT_NJOBS,
    MAIN_N_ESTIMATORS,
    MAIN_SUBSAMPLE,
    MAIN_CV,
    NUIS_N_ESTIMATORS,
)


def min_leaf_main(n: int) -> int:
    return 10



def min_leaf_nuis(n: int) -> int:
    """Nuisance森林最小叶节点样本数"""
    return int(max(5, np.ceil(0.01 * n)))


def _make_rf_nuisance(n: int, seed: int):
    leaf = min_leaf_nuis(n)

    model_t = RandomForestClassifier(
        n_estimators=NUIS_N_ESTIMATORS,
        min_samples_leaf=leaf,
        random_state=seed,
        n_jobs=DEFAULT_NJOBS
    )
    model_y = RandomForestRegressor(
        n_estimators=NUIS_N_ESTIMATORS,
        min_samples_leaf=leaf,
        random_state=seed + 1,
        n_jobs=DEFAULT_NJOBS
    )
    return model_t, model_y



def _make_misspec_nuisance(seed: int) -> Tuple[LogisticRegression, LinearRegression]:
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    reg = LinearRegression()
    return clf, reg


def fit_orf_econml(X: np.ndarray, T: np.ndarray, Y: np.ndarray, n: int,
                   seed: int, misspec: bool = False):
    try:
        from econml.orf import DMLOrthoForest
    except Exception as e:
        raise ImportError("需要 econml 才能运行 ORF。请 pip install econml") from e

    if misspec:
        model_T, model_Y = _make_misspec_nuisance(seed)
    else:
        model_T, model_Y = _make_rf_nuisance(n, seed)

    min_leaf = min_leaf_main(n)
    model = DMLOrthoForest(
        n_trees=MAIN_N_ESTIMATORS,
        subsample_ratio=MAIN_SUBSAMPLE,
        min_leaf_size=min_leaf,
        global_res_cv=MAIN_CV,
        model_T=model_T,
        model_Y=model_Y,
        discrete_treatment=True,
        random_state=seed,
        n_jobs=DEFAULT_NJOBS
    )

    model.fit(Y, T, X=X)
    return model


def fit_cf_econml(X: np.ndarray, T: np.ndarray, Y: np.ndarray, n: int, seed: int):
    try:
        from econml.grf import CausalForest
    except Exception as e:
        raise ImportError("你的 econml 版本没有 econml.grf.CausalForest。请升级 econml 或去掉 CF。") from e

    model = CausalForest(
        n_estimators=MAIN_N_ESTIMATORS,
        max_samples=MAIN_SUBSAMPLE,
        honest=True,
        inference=True,
        min_samples_leaf=min_leaf_main(n),
        random_state=seed,
        n_jobs=DEFAULT_NJOBS
    )
    model.fit(X, T, Y)
    return model



def fit_dml_cf_econml(X: np.ndarray, T: np.ndarray, Y: np.ndarray, n: int,
                      seed: int, misspec: bool = False):
    try:
        from econml.dml import CausalForestDML
    except Exception as e:
        raise ImportError("需要 econml 才能运行 DML-CF（CausalForestDML）。请 pip install econml") from e

    if misspec:
        model_t, model_y = _make_misspec_nuisance(seed)
    else:
        model_t, model_y = _make_rf_nuisance(n, seed)

    # CausalForestDML 不同版本的 econml 参数可能略有差异；这里用 signature 检查来保证兼容性
    _kwargs = dict(
        model_t=model_t,
        model_y=model_y,
        n_estimators=MAIN_N_ESTIMATORS,
        min_samples_leaf=min_leaf_main(n),
        random_state=seed,
        n_jobs=DEFAULT_NJOBS,
        inference=True,
        cv=MAIN_CV,
        discrete_treatment=True,  # ✅ 加这一行（关键）
        honest=True
    )

    if 'max_samples' in inspect.signature(CausalForestDML).parameters:
        _kwargs['max_samples'] = MAIN_SUBSAMPLE
    model = CausalForestDML(**_kwargs)
    model.fit(Y, T, X=X)
    return model



def get_method_hparams(method: str, n: int, misspec: bool = False) -> dict:
    base = dict(
        method=method,
        main_n_estimators=MAIN_N_ESTIMATORS,
        main_subsample=MAIN_SUBSAMPLE,
        main_min_leaf=min_leaf_main(n),
        main_cv=MAIN_CV,
        n_jobs=DEFAULT_NJOBS,
        nuisance_type=("linear" if misspec else "rf"),
        nuis_n_estimators=(None if misspec else NUIS_N_ESTIMATORS),
        nuis_min_leaf=(None if misspec else min_leaf_nuis(n)),
    )
    if method == "ORF":
        base.update(dict(orf_global_res_cv=MAIN_CV, orf_subsample_ratio=MAIN_SUBSAMPLE))
    if method == "CF":
        base.update(dict(cf_honest=True, cf_inference=True, cf_max_samples=MAIN_SUBSAMPLE))
    if method == "DML-CF":
        base.update(dict(dml_cv=MAIN_CV, dml_inference=True, dml_max_samples=MAIN_SUBSAMPLE))
    return base
