import numpy as np
import pytest
from scipy import sparse
from types import SimpleNamespace

from src.ingestion.best60_research.estimators import FitBudgetExceeded, fit_model, fm_objective


def test_fm_objective_gradient_matches_finite_differences():
    rng = np.random.default_rng(1312)
    X = rng.integers(0, 2, (12, 4)).astype(float)
    nuisance = np.column_stack([np.ones(12), rng.normal(size=12)])
    y = rng.integers(0, 2, 12).astype(float)
    weights = np.linspace(0.0, 1.0, 12)
    theta = rng.normal(0, 0.1, 2 + 4 + 4 * 2)
    kwargs = {"rank": 2, "penalty": 0.7, "sample_weight": weights}
    value, gradient = fm_objective(theta, X, y, nuisance, **kwargs)
    numerical = np.zeros_like(theta)
    for j in range(len(theta)):
        hi, lo = theta.copy(), theta.copy()
        hi[j] += 1e-6
        lo[j] -= 1e-6
        numerical[j] = (fm_objective(hi, X, y, nuisance, **kwargs)[0] - fm_objective(lo, X, y, nuisance, **kwargs)[0]) / 2e-6
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kind", ["nuisance", "additive", "pairs", "fm"])
def test_estimators_are_deterministic_and_predict_finite_values(kind):
    rng = np.random.default_rng(44)
    X = rng.integers(0, 2, (240, 4)).astype(float)
    nuisance = np.column_stack([np.ones(len(X)), rng.normal(size=len(X))])
    groups = np.where(np.arange(len(X)) % 2, "A", "B")
    y = rng.binomial(1, 0.05 + 0.9 * (X[:, 0] == X[:, 1])).astype(float)
    weights = np.linspace(0.25, 1.0, len(X))
    first = fit_model(X, y, nuisance=nuisance, groups=groups, kind=kind, penalty=0.01, seed=1312, sample_weight=weights)
    second = fit_model(X, y, nuisance=nuisance, groups=groups, kind=kind, penalty=0.01, seed=1312, sample_weight=weights)
    prediction = first.predict_logit(X, nuisance=nuisance, groups=groups)
    assert first.converged
    np.testing.assert_allclose(prediction, second.predict_logit(X, nuisance=nuisance, groups=groups), atol=1e-10)
    assert np.isfinite(prediction).all()


def test_additive_null_and_planted_pair_are_separated_on_new_draws():
    rng = np.random.default_rng(100)
    X_train = rng.integers(0, 2, (5000, 2)).astype(float)
    X_test = rng.integers(0, 2, (5000, 2)).astype(float)
    nuisance_train = np.ones((len(X_train), 1))
    nuisance_test = np.ones((len(X_test), 1))
    p_train = 1 / (1 + np.exp(-(-0.2 + 0.7 * X_train[:, 0] - 0.5 * X_train[:, 1] + 1.4 * X_train[:, 0] * X_train[:, 1])))
    y_train = rng.binomial(1, p_train)
    p_test = 1 / (1 + np.exp(-(-0.2 + 0.7 * X_test[:, 0] - 0.5 * X_test[:, 1] + 1.4 * X_test[:, 0] * X_test[:, 1])))
    y_test = rng.binomial(1, p_test)
    additive = fit_model(X_train, y_train, nuisance=nuisance_train, groups=np.repeat("A", len(y_train)), kind="additive", penalty=1e-4)
    pairs = fit_model(X_train, y_train, nuisance=nuisance_train, groups=np.repeat("A", len(y_train)), kind="pairs", penalty=1e-4)
    def loss(model):
        logits = model.predict_logit(X_test, nuisance=nuisance_test, groups=np.repeat("A", len(y_test)))
        return np.mean(np.logaddexp(0.0, logits) - y_test * logits)
    assert loss(pairs) < loss(additive) - 0.005


def test_pair_model_does_not_improve_a_balanced_additive_null():
    rng = np.random.default_rng(71)
    compositions = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    X_train = np.tile(np.repeat(compositions, 2, axis=0), (400, 1))
    X_test = np.tile(np.repeat(compositions, 2, axis=0), (200, 1))
    y_train = np.tile([0.0, 1.0], len(X_train) // 2)
    y_test = np.tile([0.0, 1.0], len(X_test) // 2)
    order_train, order_test = rng.permutation(len(X_train)), rng.permutation(len(X_test))
    X_train, y_train = X_train[order_train], y_train[order_train]
    X_test, y_test = X_test[order_test], y_test[order_test]
    n_train, n_test = np.ones((len(X_train), 1)), np.ones((len(X_test), 1))
    a = fit_model(X_train, y_train, nuisance=n_train, groups=None, kind="additive", penalty=1e-4)
    b = fit_model(X_train, y_train, nuisance=n_train, groups=None, kind="pairs", penalty=1e-4)
    scores = []
    for model in (a, b):
        logits = model.predict_logit(X_test, nuisance=n_test)
        scores.append(float(np.mean(np.logaddexp(0.0, logits) - y_test * logits)))
    assert scores[1] >= scores[0] - 0.005


@pytest.mark.parametrize("interaction", [0.0, 1.6], ids=["fm-additive-null", "fm-planted-interaction"])
def test_fm_null_and_planted_h1_use_independent_validation_seed_selection(interaction):
    from src.ingestion.best60_research.simulation import _block_gain

    def make_split(seed, size, event_prefix):
        rng = np.random.default_rng(seed)
        X = rng.integers(0, 2, (size, 2)).astype(float)
        groups = np.where(rng.integers(0, 2, size), "B", "A")
        nuisance = np.ones((size, 1))
        group_offset = (groups == "B") * 0.25
        logits = -0.2 + 0.65 * X[:, 0] - 0.45 * X[:, 1] + group_offset
        logits += interaction * X[:, 0] * X[:, 1]
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits))).astype(float)
        event_ids = np.asarray([f"{event_prefix}-{index // 200:03d}" for index in range(size)])
        return X, y, nuisance, groups, event_ids

    X_train, y_train, n_train, g_train, _ = make_split(101, 12_000, "train")
    X_valid, y_valid, n_valid, g_valid, _ = make_split(202, 4_000, "validation")
    X_test, y_test, n_test, g_test, event_test = make_split(303, 12_000, "test")
    additive = fit_model(
        X_train, y_train, nuisance=n_train, groups=g_train, kind="additive",
        penalty=0.02, main_effect_penalty=0.02, max_iter=500,
    )
    assert additive.converged
    additive_valid = additive.predict_logit(X_valid, nuisance=n_valid, groups=g_valid)
    additive_valid_loss = np.mean(np.logaddexp(0.0, additive_valid) - y_valid * additive_valid)

    candidates = []
    for seed in (1312, 1313, 1314):
        model = fit_model(
            X_train, y_train, nuisance=n_train, groups=g_train, kind="fm", rank=2,
            seed=seed, penalty=0.01, main_effect_penalty=0.02, max_iter=1000,
        )
        assert model.converged
        valid_logits = model.predict_logit(X_valid, nuisance=n_valid, groups=g_valid)
        validation_loss = float(np.mean(np.logaddexp(0.0, valid_logits) - y_valid * valid_logits))
        candidates.append((validation_loss, seed, model))

    # The model and seed are fixed before the independent test outcomes are scored.
    selected_valid_loss, selected_seed, selected_fm = min(candidates, key=lambda row: row[0])
    assert selected_seed in (1312, 1313, 1314)
    fm_test_logits = selected_fm.predict_logit(X_test, nuisance=n_test, groups=g_test)
    additive_test_logits = additive.predict_logit(X_test, nuisance=n_test, groups=g_test)
    fm_test_loss = np.logaddexp(0.0, fm_test_logits) - y_test * fm_test_logits
    additive_test_loss = np.logaddexp(0.0, additive_test_logits) - y_test * additive_test_logits
    gain = _block_gain(additive_test_loss - fm_test_loss, event_test, np.ones(len(y_test)), seed=1312)
    assert gain["block_count"] == 60
    assert gain["estimate"] == pytest.approx(float(np.mean(additive_test_loss - fm_test_loss)))
    if interaction:
        assert selected_valid_loss < additive_valid_loss
        assert gain["status"] == "accepted"
        assert gain["interval95"][0] > 0.0001
        assert gain["estimate"] > 0.01
    else:
        assert gain["status"] == "not_accepted"
        assert gain["interval95"][0] <= 0.0001


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_data_fails_before_optimisation(bad):
    X = np.array([[0.0], [1.0]])
    X[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        fit_model(X, np.array([0.0, 1.0]), nuisance=np.ones((2, 1)), groups=None, kind="additive")


def test_zero_weight_and_invalid_outcome_are_rejected():
    X = np.ones((4, 2))
    with pytest.raises(ValueError, match="positive total"):
        fit_model(X, np.array([0, 1, 0, 1]), nuisance=np.ones((4, 1)), groups=None, kind="nuisance", sample_weight=np.zeros(4))
    with pytest.raises(ValueError, match="binary"):
        fit_model(X, np.array([0, 1, 0, 2]), nuisance=np.ones((4, 1)), groups=None, kind="nuisance")


def test_fit_deadline_raises_explicit_budget_exception():
    X = np.tile(np.array([[0.0, 1.0], [1.0, 0.0]]), (50, 1))
    with pytest.raises(FitBudgetExceeded, match="budget"):
        fit_model(X, np.tile([0.0, 1.0], 50), nuisance=np.ones((100, 1)), groups=None,
                  kind="fm", max_seconds=0.0)


def test_nonconvergence_is_reported_on_model():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 8))
    y = rng.binomial(1, 0.5, len(X))
    model = fit_model(X, y, nuisance=np.ones((len(X), 1)), groups=None, kind="fm", max_iter=1)
    assert not model.converged
    assert model.status != 0


def test_sparse_copy_slot_design_matches_dense_fm_predictions():
    rng = np.random.default_rng(99)
    X = rng.integers(0, 2, (120, 5)).astype(float)
    nuisance = np.column_stack([np.ones(len(X)), rng.normal(size=len(X))])
    y = rng.binomial(1, 0.5, len(X))
    groups = np.repeat(["A", "B"], len(X) // 2)
    nuisance_sparse = sparse.csr_matrix(nuisance)
    dense = fit_model(X, y, nuisance=nuisance_sparse, groups=groups, kind="fm", rank=2, seed=1312, penalty=0.1)
    csr = fit_model(sparse.csr_matrix(X), y, nuisance=nuisance_sparse, groups=groups, kind="fm", rank=2, seed=1312, penalty=0.1)
    np.testing.assert_allclose(
        dense.predict_logit(X, nuisance=nuisance, groups=groups),
        csr.predict_logit(sparse.csr_matrix(X), nuisance=nuisance_sparse, groups=groups),
        atol=1e-8,
    )


def test_public_fm_objective_supports_sparse_designs_with_more_rows_than_features():
    rng = np.random.default_rng(56)
    X = sparse.csr_matrix(rng.integers(0, 2, (30, 3)).astype(float))
    nuisance = sparse.csr_matrix(np.column_stack([np.ones(30), rng.normal(size=30)]))
    theta = rng.normal(0, 0.1, 2 + 3 + 3 * 2)
    loss, gradient = fm_objective(theta, X, rng.integers(0, 2, 30), nuisance, rank=2, penalty=0.2)
    assert np.isfinite(loss)
    assert np.isfinite(gradient).all()


@pytest.mark.parametrize("kind", ["nuisance", "additive", "pairs", "fm"])
def test_private_fit_objective_gradient_includes_grouped_terms(monkeypatch, kind):
    import src.ingestion.best60_research.estimators as module

    rng = np.random.default_rng(81)
    X = rng.integers(0, 2, (18, 3)).astype(float)
    nuisance = np.column_stack([np.ones(len(X)), rng.normal(size=len(X))])
    y = rng.integers(0, 2, len(X)).astype(float)
    groups = np.asarray(["A", "B"] * 9)
    captured = {}

    def fake_minimize(fun, x0, **kwargs):
        value, gradient = fun(x0)
        numerical = np.zeros_like(x0)
        for j in range(len(x0)):
            hi, lo = x0.copy(), x0.copy()
            hi[j] += 1e-6
            lo[j] -= 1e-6
            numerical[j] = (fun(hi)[0] - fun(lo)[0]) / 2e-6
        captured["error"] = float(np.max(np.abs(gradient - numerical))) if len(x0) else 0.0
        return SimpleNamespace(x=x0, success=True, status=0, message="test", fun=value, nit=0, jac=gradient)

    monkeypatch.setattr(module, "minimize", fake_minimize)
    fit_model(
        X,
        y,
        nuisance=nuisance,
        groups=groups,
        kind=kind,
        rank=2,
        penalty=0.3,
        main_effect_penalty=0.07,
        sample_weight=np.linspace(0.1, 1, len(y)),
    )
    assert captured["error"] < 1e-6


@pytest.mark.parametrize("kind", ["nuisance", "additive", "pairs", "fm"])
@pytest.mark.parametrize("use_sparse", [False, True])
def test_fit_objective_is_invariant_to_uniform_sample_weight_rescaling(monkeypatch, kind, use_sparse):
    import src.ingestion.best60_research.estimators as module

    rng = np.random.default_rng(881)
    X = rng.integers(0, 2, (24, 3)).astype(float)
    nuisance = np.column_stack([np.ones(len(X)), rng.normal(size=len(X))])
    if use_sparse:
        X, nuisance = sparse.csr_matrix(X), sparse.csr_matrix(nuisance)
    y = rng.integers(0, 2, X.shape[0]).astype(float)
    groups = np.asarray(["A", "B"] * 12)
    weights = np.linspace(0.2, 1.0, len(y))
    captured = []

    def fake_minimize(fun, x0, **kwargs):
        theta = np.linspace(-0.23, 0.31, len(x0))
        value, gradient = fun(theta)
        captured.append((value, gradient.copy()))
        return SimpleNamespace(x=theta, success=True, status=0, message="test", fun=value, nit=0, jac=gradient)

    monkeypatch.setattr(module, "minimize", fake_minimize)
    for scale in (1.0, 17.0):
        fit_model(
            X, y, nuisance=nuisance, groups=groups, kind=kind, rank=2,
            penalty=0.3, main_effect_penalty=0.07, sample_weight=scale * weights,
        )
    np.testing.assert_allclose(captured[0][0], captured[1][0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(captured[0][1], captured[1][1], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("use_sparse", [False, True])
def test_grouped_fm_production_objective_is_weighted_mean_with_declared_penalties(monkeypatch, use_sparse):
    import src.ingestion.best60_research.estimators as module

    rng = np.random.default_rng(991)
    X_dense = rng.integers(0, 2, (20, 3)).astype(float)
    nuisance_dense = np.column_stack([np.ones(len(X_dense)), rng.normal(size=len(X_dense))])
    X = sparse.csr_matrix(X_dense) if use_sparse else X_dense
    nuisance = sparse.csr_matrix(nuisance_dense) if use_sparse else nuisance_dense
    y = rng.integers(0, 2, len(X_dense)).astype(float)
    weights = np.linspace(0.3, 1.2, len(y))
    groups = np.asarray(["A", "B"] * 10)
    rank, penalty, main_penalty = 2, 0.4, 0.08
    theta = np.linspace(-0.19, 0.27, 2 + 3 + 3 + 3 * rank)
    captured = {}

    def fake_minimize(fun, x0, **kwargs):
        value, gradient = fun(theta)
        captured["gradient"] = gradient.copy()
        return SimpleNamespace(x=theta, success=True, status=0, message="test", fun=value, nit=0, jac=gradient)

    monkeypatch.setattr(module, "minimize", fake_minimize)
    model = fit_model(
        X, y, nuisance=nuisance, groups=groups, kind="fm", rank=rank,
        penalty=penalty, main_effect_penalty=main_penalty, sample_weight=weights,
    )
    beta_n = theta[:2]
    beta_x = theta[2:5]
    beta_dev = theta[5:8]
    factors = theta[8:].reshape(3, rank)
    deviation_design = X_dense * (groups == "B")[:, None]
    latent = X_dense @ factors
    interactions = 0.5 * np.sum(latent ** 2 - (X_dense ** 2) @ (factors ** 2), axis=1)
    logits = nuisance_dense @ beta_n + X_dense @ beta_x + deviation_design @ beta_dev + interactions
    mean_log_loss = np.average(np.logaddexp(0.0, logits) - y * logits, weights=weights)
    expected = (
        mean_log_loss
        + 0.5 * module._NUISANCE_PENALTY * np.dot(beta_n[1:], beta_n[1:])
        + 0.5 * main_penalty * (np.dot(beta_x, beta_x) + np.dot(beta_dev, beta_dev))
        + 0.5 * penalty * np.sum(factors ** 2)
    )
    assert model.objective == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert model.gradient_norm == pytest.approx(np.linalg.norm(captured["gradient"]))


def test_sparse_pair_estimator_screens_training_cooccurrences():
    rng = np.random.default_rng(33)
    X = sparse.random(100, 500, density=0.01, random_state=rng, format="csr", data_rvs=lambda n: np.ones(n))
    y = rng.binomial(1, 0.5, 100)
    model = fit_model(X, y, nuisance=np.ones((len(y), 1)), groups=None, kind="pairs", penalty=0.1)
    assert model.converged
    assert len(model.pair_indices) <= 256
    assert np.isfinite(model.predict_logit(X, nuisance=np.ones((len(y), 1)))).all()


def test_group_reference_and_fit_are_row_order_invariant():
    rng = np.random.default_rng(31)
    X = rng.integers(0, 2, (180, 4)).astype(float)
    nuisance = np.column_stack([np.ones(len(X)), rng.normal(size=len(X))])
    groups = np.asarray(["Z", "A", "M"] * 60)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.2 + 0.3 * X[:, 0]))))
    order = rng.permutation(len(y))
    first = fit_model(X, y, nuisance=nuisance, groups=groups, kind="additive", seed=1312, penalty=0.2)
    second = fit_model(X[order], y[order], nuisance=nuisance[order], groups=groups[order], kind="additive", seed=1312, penalty=0.2)
    assert first.reference_group == second.reference_group == "A"
    np.testing.assert_allclose(
        first.predict_logit(X, nuisance=nuisance, groups=groups),
        second.predict_logit(X, nuisance=nuisance, groups=groups),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("kind", ["additive", "pairs", "fm"])
def test_sparse_group_storage_scales_with_observed_entries(monkeypatch, kind):
    import src.ingestion.best60_research.estimators as module

    rng = np.random.default_rng(815)
    X = sparse.csr_matrix(rng.integers(0, 2, (60, 4)).astype(float))
    groups = np.array([f"g{i % 6}" for i in range(60)])
    y = rng.binomial(1, 0.5, 60)
    storage = []
    original_hstack = module.sparse.hstack

    def observed_hstack(blocks, *args, **kwargs):
        result = original_hstack(blocks, *args, **kwargs)
        storage.append((result.nnz, np.count_nonzero(result.data)))
        return result

    monkeypatch.setattr(module.sparse, "hstack", observed_hstack)
    model = fit_model(X, y, nuisance=np.ones((60, 1)), groups=groups,
                      kind=kind, penalty=0.2, max_iter=1000)
    grouped = model._group_design(X, groups)
    expected = np.column_stack([
        X.toarray() * (groups == level)[:, None] for level in model.group_levels[1:]
    ])
    np.testing.assert_array_equal(grouped.toarray(), expected)
    assert grouped.nnz <= X.nnz
    # The real fit and prediction paths must both discard off-group zeros.
    assert storage and all(stored == nonzero for stored, nonzero in storage)
