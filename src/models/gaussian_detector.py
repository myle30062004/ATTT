"""Gaussian anomaly detection (density estimation on normal traffic).

This is *not* Gaussian Naive Bayes: no class-conditional model of attacks is
learned. The detector estimates the density p(x) of **normal** traffic only and
flags a sample as anomalous when ``log p(x) < log epsilon``.

Univariate (features assumed independent)::

    log p(x) = sum_j [ -0.5*log(2*pi*sigma_j^2) - (x_j - mu_j)^2 / (2*sigma_j^2) ]

Multivariate (full covariance Sigma, captures correlated features)::

    log p(x) = -0.5 * [ k*log(2*pi) + log|Sigma| + (x-mu)^T Sigma^-1 (x-mu) ]

Sigma is regularised as ``Sigma + lambda*I`` (ridge) so it stays positive
definite when features are collinear (singular covariance); log|Sigma| and the
Mahalanobis term are computed from a Cholesky factor for numerical stability.
Log densities are used throughout to avoid underflow of products of many small
probabilities.

The anomaly *score* returned by :meth:`score_samples` is ``-log p(x)``: higher
means less typical of normal traffic. It is **not** a probability of attack.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

LOG_2PI = np.log(2.0 * np.pi)


class UnivariateGaussianDetector:
    def __init__(self, min_variance: float = 1e-6) -> None:
        self.min_variance = min_variance
        self.mu: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.log_epsilon: float | None = None

    def fit(self, x_normal: np.ndarray) -> "UnivariateGaussianDetector":
        x = np.asarray(x_normal, dtype=np.float64)
        if len(x) < 2:
            raise ValueError("need at least 2 normal samples")
        self.mu = x.mean(0)
        self.var = np.maximum(x.var(0), self.min_variance)
        return self

    def log_density(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (-0.5 * (LOG_2PI + np.log(self.var)) - (x - self.mu) ** 2 / (2.0 * self.var)).sum(1)

    def feature_log_density(self, x: np.ndarray) -> np.ndarray:
        """Per-feature log density terms; the most negative terms explain an alert."""
        x = np.asarray(x, dtype=np.float64)
        return -0.5 * (LOG_2PI + np.log(self.var)) - (x - self.mu) ** 2 / (2.0 * self.var)

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        return -self.log_density(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.log_density(x) < self.log_epsilon).astype(int)


class MultivariateGaussianDetector:
    def __init__(self, regularization: float = 1e-3) -> None:
        self.regularization = regularization
        self.mu: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self._chol = None
        self._log_det: float = 0.0
        self.log_epsilon: float | None = None
        self.condition_number: float = np.nan
        self.added_jitter: float = 0.0

    def fit(self, x_normal: np.ndarray) -> "MultivariateGaussianDetector":
        x = np.asarray(x_normal, dtype=np.float64)
        if len(x) < 2:
            raise ValueError("need at least 2 normal samples")
        self.mu = x.mean(0)
        cov = np.cov(x, rowvar=False)
        cov = np.atleast_2d(cov)
        k = cov.shape[0]
        jitter = self.regularization
        for _ in range(8):  # increase ridge until Cholesky succeeds
            try:
                self._chol = cho_factor(cov + jitter * np.eye(k), lower=True)
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        else:
            raise np.linalg.LinAlgError("covariance not positive definite even after regularisation")
        self.added_jitter = jitter
        self.cov = cov + jitter * np.eye(k)
        self._log_det = 2.0 * np.log(np.diag(self._chol[0])).sum()
        self.condition_number = float(np.linalg.cond(self.cov))
        return self

    def mahalanobis_sq(self, x: np.ndarray) -> np.ndarray:
        d = np.asarray(x, dtype=np.float64) - self.mu
        return np.einsum("ij,ij->i", d, cho_solve(self._chol, d.T).T)

    def log_density(self, x: np.ndarray) -> np.ndarray:
        k = self.mu.shape[0]
        return -0.5 * (k * LOG_2PI + self._log_det + self.mahalanobis_sq(x))

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        return -self.log_density(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.log_density(x) < self.log_epsilon).astype(int)


def select_epsilon(log_density: np.ndarray, y_true: np.ndarray, metric: str = "f1",
                   n_candidates: int = 1000, max_fpr: float | None = None) -> dict:
    """Choose log(epsilon) on a labelled validation set (1 = attack, 0 = normal).

    Candidates are quantiles of the validation log densities; a sample is
    anomalous when ``log p(x) < log_epsilon``. ``metric`` is ``f1`` or ``f2``
    (recall-weighted). With ``max_fpr``, only thresholds whose false-positive
    rate on normal validation traffic is <= max_fpr are considered.
    """
    y_true = np.asarray(y_true).astype(int)
    log_density = np.asarray(log_density, dtype=np.float64)
    attack = np.sort(log_density[y_true == 1])
    normal = np.sort(log_density[y_true == 0])
    candidates = np.unique(np.quantile(log_density, np.linspace(0.0005, 0.9995, n_candidates)))
    tp = np.searchsorted(attack, candidates, side="left").astype(np.float64)
    fp = np.searchsorted(normal, candidates, side="left").astype(np.float64)
    fn = len(attack) - tp
    beta2 = 1.0 if metric == "f1" else 4.0
    denom = (1 + beta2) * tp + beta2 * fn + fp
    score = np.divide((1 + beta2) * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    fpr = fp / max(len(normal), 1)
    if max_fpr is not None:
        score = np.where(fpr <= max_fpr, score, -1.0)
    i = int(np.argmax(score))
    return {"log_epsilon": float(candidates[i]), "score": float(score[i]), "fpr": float(fpr[i]),
            "recall": float(tp[i] / max(len(attack), 1)),
            "precision": float(tp[i] / max(tp[i] + fp[i], 1)), "metric": metric}


class GaussianMixtureDetector:
    """Mixture of K multivariate Gaussians fitted (EM) on normal traffic only.

        p(x) = sum_k pi_k * N(x; mu_k, Sigma_k)

    Normal traffic is multi-modal (DNS, HTTPS, RDP, ... each form their own
    cluster); a single Gaussian puts most of its mass *between* those clusters,
    where short tool-generated attack flows also lie. A mixture keeps the
    density-threshold idea (anomaly if log p(x) < log epsilon) while modelling
    several modes. ``reg_covar`` adds a ridge to every component covariance.
    ``mu``/``var`` (overall mean/variance of the training data) are kept for
    per-feature z-score explanations.
    """

    def __init__(self, n_components: int = 8, reg_covar: float = 1e-3, seed: int = 42, max_iter: int = 200) -> None:
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.seed = seed
        self.max_iter = max_iter
        self.model = None
        self.mu: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.log_epsilon: float | None = None

    def fit(self, x_normal: np.ndarray) -> "GaussianMixtureDetector":
        from sklearn.mixture import GaussianMixture

        x = np.asarray(x_normal, dtype=np.float64)
        if len(x) < self.n_components * 2:
            raise ValueError("not enough normal samples for the mixture")
        self.model = GaussianMixture(self.n_components, covariance_type="full", reg_covar=self.reg_covar,
                                     random_state=self.seed, max_iter=self.max_iter).fit(x)
        self.mu = x.mean(0)
        self.var = np.maximum(x.var(0), 1e-6)
        return self

    def log_density(self, x: np.ndarray) -> np.ndarray:
        return self.model.score_samples(np.asarray(x, dtype=np.float64))

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        return -self.log_density(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.log_density(x) < self.log_epsilon).astype(int)


def threshold_at_fpr(normal_scores: np.ndarray, target_fpr: float) -> float:
    """Threshold on a normality score (higher = more normal) so that a fraction
    ``target_fpr`` of the given normal samples falls below it."""
    return float(np.quantile(np.asarray(normal_scores, dtype=np.float64), target_fpr))
