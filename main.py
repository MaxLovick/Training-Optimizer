"""
Personalized Fitness Optimizer
=========================================================

Cross-domain fitness engine combining running, strength training, and
user-defined daily metrics to predict goal achievement and optimize
training plans.

Core statistical methods (32 total):
  - Banister impulse-response model (two-component)
  - Bayesian linear regression for improvement trajectories
  - Lagged cross-correlation & Granger causality
  - Thompson Sampling contextual bandit
  - Ridge regression for readiness personalization
  - Logistic regression for injury risk
  - BOCPD for changepoint detection
  - Shapley-based race post-mortem
  - ...and more (see planning doc for full list)

Environment:
  Python 3.12.8 | NumPy 1.26.4 | Pandas 3.0.1 | Sklearn 1.8.0
  SciPy (system) | Matplotlib 3.10.8

Usage:
  Import classes/functions as needed. See section docstrings for details.
  All public methods include type hints and parameter documentation.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum, auto
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats


"""
====================================================================
ENUMS & CONSTANTS
====================================================================
Domain enums (workout types, readiness bands, overreaching levels,
metric data types, confidence levels) plus default-parameter dicts that
seed personalized models before user data is sufficient to override
them. BANISTER_DEFAULTS / BANISTER_BOUNDS gate the impulse-response fit;
DEFAULT_READINESS_WEIGHTS seeds the readiness composite;
INJURY_RISK_COEFFICIENTS holds the population-level logistic
coefficients used until enough personal data exists to refit.
"""

class GoalType(Enum):
    ENDURANCE = auto()
    STRENGTH = auto()
    BODY_COMP = auto()
    POWER = auto()
    FLEXIBILITY = auto()
    CUSTOM = auto()


class RunWorkoutType(Enum):
    EASY = auto()
    INTERVALS = auto()
    FARTLEK = auto()
    TEMPO = auto()
    LONG_RUN = auto()
    HILL_REPEATS = auto()
    RACE = auto()
    TIME_TRIAL = auto()
    CUSTOM = auto()


class MetricDataType(Enum):
    NUMERIC = auto()
    SCALE_1_10 = auto()
    BOOLEAN = auto()
    TIME_DURATION = auto()


class ReadinessBand(Enum):
    GREEN = "Go hard"
    YELLOW = "Moderate session"
    ORANGE = "Easy / active recovery"
    RED = "Rest day recommended"


class OverreachingLevel(Enum):
    NORMAL = "Normal"
    FUNCTIONAL = "Functional overreaching"
    NON_FUNCTIONAL = "Non-functional overreaching"


class ConfidenceLevel(Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


BANISTER_DEFAULTS = {
    "tau_1": 42.0,
    "tau_2": 12.0,
    "k_1": 1.0,
    "k_2": 2.0,
    "p_0": 100.0,
}

BANISTER_BOUNDS = {
    "tau_1": (20.0, 80.0),
    "tau_2": (5.0, 25.0),
    "k_1": (0.01, 10.0),
    "k_2": (0.01, 10.0),
    "p_0": (0.0, 500.0),
}

DEFAULT_READINESS_WEIGHTS = {
    "sleep": 0.25,
    "fatigue": 0.25,
    "days_since_hard": 0.15,
    "acwr": 0.15,
    "stress": 0.10,
    "soreness": 0.10,
}

CONFLICT_MATRIX = {
    (GoalType.ENDURANCE, GoalType.STRENGTH): 2,
    (GoalType.ENDURANCE, GoalType.POWER): 2,
    (GoalType.ENDURANCE, GoalType.BODY_COMP): 3,
    (GoalType.STRENGTH, GoalType.POWER): 1,
    (GoalType.STRENGTH, GoalType.BODY_COMP): 1,
    (GoalType.POWER, GoalType.BODY_COMP): 1,
}

INJURY_RISK_COEFFICIENTS = {
    "intercept": -4.5,
    "acwr": 1.8,
    "monotony": 0.9,
    "strain": 0.0004,
    "fatigue": 0.6,
    "soreness": 0.5,
    "readiness_trend": -0.3,
}


"""
====================================================================
DATA MODELS
====================================================================
Plain dataclasses representing the user-facing inputs that flow into
every analytics path: Goal, RunningSession / LiftingSession / Lifting-
Exercise, MetricEntry / CustomMetricDefinition, Assessment. These are
the public schema; the analytics classes downstream consume them but
never mutate fields except `trimp` and `tonnage` (computed during
ingestion).
"""

@dataclass
class Goal:
    """Section 1 — User goal with target and milestones."""
    goal_id: str
    goal_type: GoalType
    description: str
    target_value: float
    current_value: float
    target_date: date
    created_date: date = field(default_factory=date.today)
    milestones: list[dict] = field(default_factory=list)
    conflict_tags: list[GoalType] = field(default_factory=list)


@dataclass
class RunningSession:
    """Section 2 — Running training log entry."""
    session_id: str
    session_date: date
    workout_type: RunWorkoutType
    duration_minutes: float
    distance_km: Optional[float] = None
    avg_pace_min_per_km: Optional[float] = None
    avg_heart_rate: Optional[int] = None
    max_heart_rate: Optional[int] = None
    rpe: Optional[float] = None
    intervals: Optional[list[dict]] = None
    notes: Optional[str] = None
    trimp: Optional[float] = None


@dataclass
class LiftingSession:
    """Section 3 — Lifting training log entry."""
    session_id: str
    session_date: date
    exercises: list[LiftingExercise]
    session_rpe: Optional[float] = None
    notes: Optional[str] = None
    tonnage: Optional[float] = None


@dataclass
class LiftingExercise:
    """Single exercise within a lifting session."""
    exercise_name: str
    muscle_groups: list[str]
    sets: int
    reps_per_set: list[int]
    weight: float
    rest_seconds: Optional[float] = None
    rpe: Optional[float] = None
    rir: Optional[int] = None
    tempo: Optional[str] = None


@dataclass
class CustomMetricDefinition:
    """Section 4 — User-defined daily metric schema."""
    metric_id: str
    name: str
    data_type: MetricDataType
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    unit: Optional[str] = None


@dataclass
class MetricEntry:
    """Single daily metric data point."""
    metric_id: str
    entry_date: date
    value: float
    entry_time: Optional[datetime] = None


@dataclass
class Assessment:
    """Section 5 — Performance benchmark (race, time trial, 1RM test)."""
    assessment_id: str
    assessment_date: date
    domain: str
    raw_value: float
    raw_unit: str
    assessment_type: str
    normalized_score: Optional[float] = None
    z_score: Optional[float] = None


"""
====================================================================
SECTION 1 — GOAL SETTING & MILESTONE GENERATION
====================================================================
GoalEngine produces dated intermediate targets between a goal's
current value and its target value. Milestone spacing follows a
log-improvement curve fit to the user's own assessment history when
enough exists; otherwise it falls back to a population log curve.
The fit divides by `log_model(1) - popt[1]` (the full t=0 → t=1 span),
not `fitted[-1] - fitted[0]`, so the first milestone gets a non-zero
fraction of progress (an earlier bug zeroed it out by subtracting the
first internal milestone's value).
"""

class GoalEngine:
    """
    Generates milestones using logarithmic spacing to model diminishing
    returns in athletic improvement.

    Method: Linear interpolation with diminishing-returns curve fitting.
    If historical assessments exist, fits improvement(t) = a * ln(t + 1) + b
    to the user's actual improvement rate.
    """

    @staticmethod
    def generate_milestones(
        goal: Goal,
        n_milestones: int = 4,
        assessment_history: Optional[list[Assessment]] = None,
    ) -> list[dict]:
        """
        Generate intermediate milestones between current and target value.

        Parameters
        ----------
        goal : Goal
        n_milestones : int
            Number of intermediate milestones (excluding start/end).
        assessment_history : list[Assessment], optional
            Prior assessments for curve fitting. If None or < 3,
            uses default logarithmic spacing.

        Returns
        -------
        list[dict]
            Each dict: {"date": date, "target_value": float, "label": str}
        """
        total_days = (goal.target_date - goal.created_date).days
        if total_days <= 0:
            return []

        improvement_needed = goal.target_value - goal.current_value

        if assessment_history and len(assessment_history) >= 3:
            fractions = GoalEngine._fit_personal_curve(
                assessment_history, n_milestones
            )
        else:
            raw = np.log(np.linspace(1, np.e, n_milestones + 2))
            fractions = (raw - raw[0]) / (raw[-1] - raw[0])
            fractions = fractions[1:-1]

        milestones = []
        for i, frac in enumerate(fractions):
            ms_date = goal.created_date + timedelta(
                days=int(frac * total_days)
            )
            ms_value = goal.current_value + frac * improvement_needed
            milestones.append({
                "date": ms_date,
                "target_value": round(ms_value, 2),
                "label": f"Milestone {i + 1}",
            })

        goal.milestones = milestones
        return milestones

    @staticmethod
    def _fit_personal_curve(
        assessments: list[Assessment], n_milestones: int
    ) -> np.ndarray:
        """Fit a * ln(t + 1) + b to assessment history, return milestone fractions."""
        days = np.array([
            (a.assessment_date - assessments[0].assessment_date).days
            for a in assessments
        ], dtype=float)
        scores = np.array([a.normalized_score or a.raw_value for a in assessments])

        if days[-1] == 0:
            raw = np.log(np.linspace(1, np.e, n_milestones + 2))
            fracs = (raw - raw[0]) / (raw[-1] - raw[0])
            return fracs[1:-1]

        days_norm = days / days[-1]

        try:
            def log_model(t, a, b):
                return a * np.log(t + 1) + b

            popt, _ = optimize.curve_fit(
                log_model, days_norm, scores, p0=[1.0, scores[0]], maxfev=5000
            )
            t_milestones = np.linspace(0, 1, n_milestones + 2)[1:-1]
            fitted = log_model(t_milestones, *popt)
            baseline = popt[1]
            total_improvement = log_model(1.0, *popt) - baseline
            if total_improvement == 0:
                return np.linspace(0, 1, n_milestones + 2)[1:-1]
            fracs = (fitted - baseline) / total_improvement
            return np.clip(fracs, 0, 1)
        except (RuntimeError, ValueError):
            raw = np.log(np.linspace(1, np.e, n_milestones + 2))
            fracs = (raw - raw[0]) / (raw[-1] - raw[0])
            return fracs[1:-1]


"""
====================================================================
SECTIONS 2 & 3 — TRAINING LOAD QUANTIFICATION
====================================================================
TrainingLoad converts raw sessions into single load numbers:
  - compute_trimp: HR-weighted Banister TRIMP if HR data is present,
    else RPE × duration. The resting-HR anchor is personalized: callers
    can pass `hr_rest`; FitnessOptimizer estimates one from easy-session
    avg HRs once 5+ are logged (5th percentile minus ~10 bpm) and
    falls back to 60 only when no profile and no history exist.
  - compute_tonnage: sets × reps × weight, RPE-scaled when session_rpe
    is present.
"""

class TrainingLoad:
    """
    Computes TRIMP (running) and tonnage (lifting) for each session.

    TRIMP = duration * HR_factor (or RPE * duration as proxy).
    Tonnage = sum(sets * reps * weight) per session.
    """

    @staticmethod
    def compute_trimp(
        session: RunningSession, hr_rest: Optional[int] = None
    ) -> float:
        """
        Compute Training Impulse (TRIMP) for a running session.

        Uses HR-based exponential weighting if HR data available,
        otherwise falls back to RPE * duration proxy.

        Parameters
        ----------
        session : RunningSession
        hr_rest : int, optional
            Personalized resting HR. Falls back to 60 bpm population default
            when not provided. Callers (e.g. FitnessOptimizer) should pass an
            estimated value derived from low-effort sessions.

        Returns
        -------
        float
            TRIMP value (arbitrary units, higher = more load).
        """
        if session.avg_heart_rate and session.max_heart_rate:
            hr_max = session.max_heart_rate
            hr_avg = session.avg_heart_rate
            if hr_rest is None:
                hr_rest = 60
            hr_ratio = (hr_avg - hr_rest) / (hr_max - hr_rest)
            hr_ratio = np.clip(hr_ratio, 0, 1)
            b = 1.80
            hr_factor = hr_ratio * (0.64 * np.exp(b * hr_ratio))
            trimp = session.duration_minutes * hr_factor
        elif session.rpe is not None:
            trimp = session.rpe * session.duration_minutes
        else:
            trimp = 5.0 * session.duration_minutes

        session.trimp = round(trimp, 2)
        return session.trimp

    @staticmethod
    def compute_tonnage(session: LiftingSession) -> float:
        """
        Compute total tonnage for a lifting session.

        Tonnage = sum(sets * reps * weight) across all exercises.
        Optionally weighted by session RPE.

        Parameters
        ----------
        session : LiftingSession

        Returns
        -------
        float
            Total tonnage in weight units (lbs or kg, matches input).
        """
        total = 0.0
        for ex in session.exercises:
            for rep_count in ex.reps_per_set:
                total += rep_count * ex.weight
            if len(ex.reps_per_set) < ex.sets:
                last_rep = ex.reps_per_set[-1] if ex.reps_per_set else 0
                remaining = ex.sets - len(ex.reps_per_set)
                total += remaining * last_rep * ex.weight

        if session.session_rpe is not None:
            rpe_factor = session.session_rpe / 10.0
            total *= (0.5 + 0.5 * rpe_factor)

        session.tonnage = round(total, 2)
        return session.tonnage


r"""
====================================================================
SECTION 5 — ASSESSMENT NORMALIZATION
====================================================================
AssessmentNormalizer maps raw performance numbers to a comparable
scale: VDOT for running times, Epley 1RM for lifts. Rep scheme is
parsed with a regex (\d+RM) anywhere in the assessment_type string, so
formats like "1RM_squat", "squat_3RM", and "5 RM bench" all resolve to
the correct rep count rather than silently falling back to 1.
"""

class AssessmentNormalizer:
    """
    Normalizes assessments to comparable scales using:
    - VDOT (Jack Daniels) for running
    - Epley formula for lifting (any rep max → estimated 1RM)
    - Z-scores for cross-domain comparison
    """

    @staticmethod
    def compute_vdot(distance_meters: float, time_seconds: float) -> float:
        """
        Estimate VDOT from race distance and time.

        Uses the Jack Daniels / Gilbert formula approximation.

        Parameters
        ----------
        distance_meters : float
        time_seconds : float

        Returns
        -------
        float
            VDOT score (higher = fitter).
        """
        t = time_seconds / 60.0
        d = distance_meters

        velocity = d / t
        vo2 = -4.60 + 0.182258 * velocity + 0.000104 * velocity**2

        pct_max = (
            0.8 + 0.1894393 * np.exp(-0.012778 * t)
            + 0.2989558 * np.exp(-0.1932605 * t)
        )

        vdot = vo2 / pct_max
        return round(vdot, 2)

    @staticmethod
    def epley_1rm(weight: float, reps: int) -> float:
        """
        Estimate 1RM from weight and reps using Epley formula.

        1RM = weight * (1 + reps / 30)

        Parameters
        ----------
        weight : float
        reps : int
            Must be > 0.

        Returns
        -------
        float
            Estimated 1RM.
        """
        if reps <= 0:
            raise ValueError("Reps must be > 0")
        if reps == 1:
            return float(weight)
        return round(weight * (1 + reps / 30.0), 2)

    @staticmethod
    def normalize_assessment(
        assessment: Assessment,
        history: Optional[list[Assessment]] = None,
    ) -> Assessment:
        """
        Compute normalized_score and z_score for an assessment.

        Parameters
        ----------
        assessment : Assessment
        history : list[Assessment], optional
            Prior assessments of same type for z-score computation.

        Returns
        -------
        Assessment
            Updated in place with normalized_score and z_score.
        """
        if assessment.domain == "running":
            distance_map = {
                "5K": 5000, "10K": 10000, "half_marathon": 21097.5,
                "marathon": 42195, "mile": 1609.34, "1500m": 1500,
                "800m": 800, "400m": 400,
            }
            dist = distance_map.get(assessment.assessment_type)
            if dist and assessment.raw_unit == "seconds":
                assessment.normalized_score = AssessmentNormalizer.compute_vdot(
                    dist, assessment.raw_value
                )
        elif assessment.domain == "lifting":
            match = re.search(r"(\d+)\s*RM", assessment.assessment_type)
            reps = int(match.group(1)) if match else 1
            assessment.normalized_score = AssessmentNormalizer.epley_1rm(
                assessment.raw_value, reps
            )

        if history and len(history) >= 2:
            scores = [
                a.normalized_score for a in history
                if a.normalized_score is not None
            ]
            if scores:
                mean = np.mean(scores)
                std = np.std(scores, ddof=1)
                if std > 0 and assessment.normalized_score is not None:
                    assessment.z_score = round(
                        (assessment.normalized_score - mean) / std, 3
                    )

        return assessment


"""
====================================================================
SECTION 6 — PREDICTION & PROJECTION ENGINE
====================================================================
PredictionEngine fits a Bayesian-blended linear regression to
normalized assessment scores. Implementation contract:
  - Filters out assessments with normalized_score=None first, *then*
    sorts the remaining set by assessment_date — so the "first"
    assessment used as the day-zero reference is the earliest *included*
    one, not whatever happened to land at index 0.
  - Stores `self.base_date` so callers can pass the matching date to
    `predict()` instead of guessing.
  - Slope is a precision-weighted blend of the data fit and a sports-
    science prior, weight w = min(1, n/15).
"""

class PredictionEngine:
    """
    Bayesian linear regression on normalized assessment scores to project
    improvement trajectory and goal likelihood.

    Produces confidence intervals rather than point estimates.
    Minimum data requirement: 4 assessments.

    For v1: uses scipy-based OLS with bootstrapped confidence intervals.
    For full Bayesian: would use PyMC / Stan (see planning doc).
    """

    MIN_ASSESSMENTS = 4

    def __init__(
        self,
        prior_slope_mean: float = 0.02,
        prior_slope_std: float = 0.01,
    ):
        """
        Parameters
        ----------
        prior_slope_mean : float
            Expected improvement rate per day (from sports science).
        prior_slope_std : float
            Uncertainty in prior slope.
        """
        self.prior_slope_mean = prior_slope_mean
        self.prior_slope_std = prior_slope_std
        self._fitted = False
        self.slope = None
        self.intercept = None
        self.slope_std = None
        self.residual_std = None
        self.base_date: Optional[date] = None

    def fit(self, assessments: list[Assessment]) -> dict:
        """
        Fit Bayesian-inspired linear regression to assessment history.

        Uses weighted blending of prior and data-driven estimates.

        Parameters
        ----------
        assessments : list[Assessment]
            Must have normalized_score set. Sorted by date.

        Returns
        -------
        dict
            {"slope", "intercept", "slope_std", "r_squared",
             "n_assessments", "personalization_pct"}
        """
        if len(assessments) < self.MIN_ASSESSMENTS:
            raise ValueError(
                f"Need >= {self.MIN_ASSESSMENTS} assessments, got {len(assessments)}"
            )

        usable = sorted(
            (a for a in assessments if a.normalized_score is not None),
            key=lambda a: a.assessment_date,
        )

        if len(usable) < self.MIN_ASSESSMENTS:
            raise ValueError("Not enough assessments with normalized scores")

        base_date = usable[0].assessment_date
        scores = np.array([a.normalized_score for a in usable])
        days = np.array(
            [(a.assessment_date - base_date).days for a in usable], dtype=float
        )
        self.base_date = base_date

        slope_data, intercept_data, r_value, p_value, se_slope = stats.linregress(
            days, scores
        )

        n = len(scores)
        N_threshold = 15
        w = min(1.0, n / N_threshold)

        if se_slope > 0:
            data_precision = 1.0 / se_slope**2
        else:
            data_precision = 1e6
        prior_precision = 1.0 / self.prior_slope_std**2

        combined_precision = w * data_precision + (1 - w) * prior_precision
        self.slope = (
            (w * data_precision * slope_data
             + (1 - w) * prior_precision * self.prior_slope_mean)
            / combined_precision
        )
        self.slope_std = 1.0 / np.sqrt(combined_precision)
        self.intercept = intercept_data
        self.residual_std = np.std(scores - (self.slope * days + self.intercept), ddof=2)
        self._fitted = True

        return {
            "slope": round(self.slope, 6),
            "intercept": round(self.intercept, 4),
            "slope_std": round(self.slope_std, 6),
            "r_squared": round(r_value**2, 4),
            "n_assessments": n,
            "personalization_pct": round(w * 100, 1),
        }

    def predict(
        self,
        target_date: date,
        base_date: date,
        target_value: float,
        confidence_levels: list[float] = None,
    ) -> dict:
        """
        Predict performance at target_date with confidence intervals.

        Parameters
        ----------
        target_date : date
        base_date : date
            Reference date (first assessment).
        target_value : float
            Goal target (normalized score).
        confidence_levels : list[float]
            Confidence levels for intervals (default [0.50, 0.70, 0.90]).

        Returns
        -------
        dict
            {"predicted_mean", "intervals": {level: (low, high)},
             "goal_probability": float}
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        if confidence_levels is None:
            confidence_levels = [0.50, 0.70, 0.90]

        days_ahead = (target_date - base_date).days
        predicted_mean = self.slope * days_ahead + self.intercept

        prediction_std = np.sqrt(
            self.slope_std**2 * days_ahead**2 + self.residual_std**2
        )

        intervals = {}
        for level in confidence_levels:
            z = stats.norm.ppf(0.5 + level / 2)
            low = predicted_mean - z * prediction_std
            high = predicted_mean + z * prediction_std
            intervals[level] = (round(low, 3), round(high, 3))

        if prediction_std > 0:
            goal_prob = 1.0 - stats.norm.cdf(
                target_value, loc=predicted_mean, scale=prediction_std
            )
        else:
            goal_prob = 1.0 if predicted_mean >= target_value else 0.0

        return {
            "predicted_mean": round(predicted_mean, 3),
            "prediction_std": round(prediction_std, 3),
            "intervals": intervals,
            "goal_probability": round(goal_prob, 4),
            "days_ahead": days_ahead,
        }


"""
====================================================================
SECTION 7 — CORRELATION / INSIGHT ENGINE
====================================================================
CorrelationEngine discovers temporal links between daily metrics and
performance:
  - lagged_cross_correlation: tests Pearson r at every lag from 0 to
    max_lag, returning the strongest. Skips lags where either window is
    near-constant (std < 1e-10) to avoid pearsonr blowing up.
  - adf_stationarity_test: uses statsmodels.tsa.stattools.adfuller when
    available; falls back to a variance-ratio heuristic otherwise.
  - granger_causality_test: F-test on whether past X improves Y's
    prediction beyond Y-only. Returns {"error": ...} when no lag
    produces an estimable model (instead of a hardcoded p_value=1).
  - benjamini_hochberg: FDR correction across many parallel tests.
"""

class CorrelationEngine:
    """
    Discovers temporal correlations between daily metrics and performance.

    Methods:
      7a. Lagged Pearson cross-correlation (lags 0-14 days)
      7b. Granger causality test (with ADF stationarity check)
      7c. Benjamini-Hochberg FDR correction for multiple testing
    """

    MIN_DATA_POINTS = 30

    @staticmethod
    def lagged_cross_correlation(
        metric_series: np.ndarray,
        performance_series: np.ndarray,
        max_lag: int = 14,
    ) -> dict:
        """
        Section 7a — Compute Pearson correlation at lags 0 to max_lag.

        Parameters
        ----------
        metric_series : np.ndarray
            Daily metric values (aligned by date index).
        performance_series : np.ndarray
            Performance scores at corresponding dates.
        max_lag : int

        Returns
        -------
        dict
            {"correlations": [(lag, r, p_value), ...],
             "best_lag": int, "best_r": float, "best_p": float}
        """
        n = len(metric_series)
        if n < CorrelationEngine.MIN_DATA_POINTS:
            return {"error": f"Need >= {CorrelationEngine.MIN_DATA_POINTS} points, got {n}"}

        results = []
        for lag in range(max_lag + 1):
            if lag >= n - 2:
                break
            x = metric_series[:n - lag]
            y = performance_series[lag:]
            min_len = min(len(x), len(y))
            x, y = x[:min_len], y[:min_len]

            if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                results.append((lag, 0.0, 1.0))
                continue

            r, p = stats.pearsonr(x, y)
            results.append((lag, round(r, 4), round(p, 6)))

        if not results:
            return {"error": "No valid lag computations"}

        best = max(results, key=lambda t: abs(t[1]))
        return {
            "correlations": results,
            "best_lag": best[0],
            "best_r": best[1],
            "best_p": best[2],
        }

    @staticmethod
    def adf_stationarity_test(series: np.ndarray) -> dict:
        """
        Augmented Dickey-Fuller test for stationarity.

        Uses statsmodels.tsa.stattools.adfuller when available; otherwise
        falls back to a variance-ratio heuristic.

        Returns
        -------
        dict
            With statsmodels: {"adf_statistic", "p_value", "is_stationary"}.
            Heuristic fallback: {"var_ratio", "is_stationary", "note"}.
        """
        if len(series) < 10:
            return {"error": "Need >= 10 data points for ADF"}

        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            adfuller = None

        if adfuller is not None:
            adf_stat, p_value, *_ = adfuller(series, autolag="AIC")
            return {
                "adf_statistic": round(float(adf_stat), 4),
                "p_value": round(float(p_value), 5),
                "is_stationary": bool(p_value < 0.05),
            }

        diffs = np.diff(series)
        if len(diffs) < 5:
            return {"is_stationary": True, "note": "too short to assess"}

        var_ratio = np.var(diffs) / (np.var(series) + 1e-10)
        return {
            "var_ratio": round(var_ratio, 4),
            "is_stationary": bool(var_ratio > 0.5),
            "note": "Heuristic fallback (install statsmodels for true ADF).",
        }

    @staticmethod
    def granger_causality_test(
        x_series: np.ndarray,
        y_series: np.ndarray,
        max_lag: int = 7,
    ) -> dict:
        """
        Section 7b — Simplified Granger causality test.

        Tests whether past values of X improve prediction of Y beyond
        what past values of Y alone provide.

        Parameters
        ----------
        x_series : np.ndarray
            Potential cause series.
        y_series : np.ndarray
            Outcome series (aligned dates).
        max_lag : int

        Returns
        -------
        dict
            {"lags_tested": int, "best_lag": int, "f_stat": float,
             "p_value": float, "variance_explained_pct": float}
        """
        n = len(y_series)
        if n < CorrelationEngine.MIN_DATA_POINTS:
            return {"error": f"Need >= {CorrelationEngine.MIN_DATA_POINTS} points"}

        best_result = {"p_value": 1.0, "f_stat": 0.0, "lag": 1, "var_exp": 0.0}
        had_valid_iteration = False

        for lag in range(1, max_lag + 1):
            if n - lag < 10:
                break

            Y = y_series[lag:]
            Y_lags = np.column_stack([y_series[lag - j - 1: n - j - 1] for j in range(lag)])

            X_lags = np.column_stack([x_series[lag - j - 1: n - j - 1] for j in range(lag)])

            min_len = min(len(Y), Y_lags.shape[0], X_lags.shape[0])
            Y = Y[:min_len]
            Y_lags = Y_lags[:min_len]
            X_lags = X_lags[:min_len]

            try:
                Y_lags_with_const = np.column_stack([np.ones(min_len), Y_lags])
                beta_r = np.linalg.lstsq(Y_lags_with_const, Y, rcond=None)[0]
                resid_r = Y - Y_lags_with_const @ beta_r
                ssr_r = np.sum(resid_r**2)

                full_X = np.column_stack([Y_lags_with_const, X_lags])
                beta_u = np.linalg.lstsq(full_X, Y, rcond=None)[0]
                resid_u = Y - full_X @ beta_u
                ssr_u = np.sum(resid_u**2)

                df_extra = lag
                df_resid = min_len - 2 * lag - 1
                if df_resid <= 0 or ssr_u <= 0:
                    continue

                f_stat = ((ssr_r - ssr_u) / df_extra) / (ssr_u / df_resid)
                p_value = 1.0 - stats.f.cdf(f_stat, df_extra, df_resid)
                var_exp = (ssr_r - ssr_u) / (ssr_r + 1e-10) * 100
                had_valid_iteration = True

                if p_value < best_result["p_value"]:
                    best_result = {
                        "p_value": round(p_value, 6),
                        "f_stat": round(f_stat, 3),
                        "lag": lag,
                        "var_exp": round(var_exp, 2),
                    }
            except np.linalg.LinAlgError:
                continue

        if not had_valid_iteration:
            return {"error": "No lag produced an estimable model"}

        return {
            "lags_tested": max_lag,
            "best_lag": best_result["lag"],
            "f_stat": best_result["f_stat"],
            "p_value": best_result["p_value"],
            "variance_explained_pct": best_result["var_exp"],
            "is_significant": best_result["p_value"] < 0.05 and best_result["var_exp"] >= 5.0,
        }

    @staticmethod
    def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
        """
        Section 7c — Benjamini-Hochberg FDR correction.

        Parameters
        ----------
        p_values : list[float]
        alpha : float
            FDR threshold.

        Returns
        -------
        list[bool]
            True for p-values that survive correction (same order as input).
        """
        m = len(p_values)
        if m == 0:
            return []
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        results = [False] * m
        largest_k = 0
        for rank, (_, p) in enumerate(indexed, 1):
            if p <= (rank / m) * alpha:
                largest_k = rank
        for rank, (orig_idx, _) in enumerate(indexed, 1):
            if rank <= largest_k:
                results[orig_idx] = True
        return results

    @staticmethod
    def generate_insight(
        metric_name: str,
        best_lag: int,
        best_r: float,
        best_p: float,
        granger_significant: bool,
    ) -> Optional[str]:
        """Generate natural language insight from correlation results."""
        if best_p > 0.05 or abs(best_r) < 0.3:
            return None

        direction = "positively" if best_r > 0 else "negatively"
        strength = "strongly" if abs(best_r) > 0.6 else "moderately"
        causal_note = " (likely contributor)" if granger_significant else ""

        lag_text = f"{best_lag} day{'s' if best_lag != 1 else ''}" if best_lag > 0 else "same day"

        return (
            f"{metric_name} {strength} {direction} correlates with performance "
            f"at a {lag_text} lag (r = {best_r:.2f}){causal_note}."
        )


"""
====================================================================
SECTION 9 — BANISTER IMPULSE-RESPONSE MODEL & SIMULATOR
====================================================================
BanisterModel implements P(t) = p_0 + k_1·Fitness(t) − k_2·Fatigue(t),
with fitness/fatigue as exponential-decay sums over past loads
(time constants tau_1 > tau_2).

fit_parameters: reparameterized as (p_0, k_1, k_2, tau_2, delta) where
tau_1 = tau_2 + delta and delta > 0 by construction. This enforces the
ordering during the optimization rather than swapping parameters after
the fit (which would decouple k_1/k_2 from the time constants they
were fit against).

simulate_session: applies adaptation_quality `q` to the proposed load
before inserting it (effective_load = proposed_load * q), so the
returned curves reflect the same q the dashboard reports — wasted
training shouldn't credit full fitness gains.

Also exposes ACWR computation, taper optimization, and per-session
contribution decomposition for visualization.
"""

class BanisterModel:
    """
    Section 9a — Two-component impulse-response model (Banister 1975).

    Performance(t) = p0 + k1 * Fitness(t) - k2 * Fatigue(t)

    Where Fitness and Fatigue are superpositions of exponentially
    decaying contributions from each training session.

    Includes:
      9b. Levenberg-Marquardt parameter fitting
      9c. Cumulative lag visualization data
      9d. Adaptation quality multiplier (sigmoid discount)
      9e. Overreaching detection
      9f. Taper optimization
    """

    def __init__(
        self,
        tau_1: float = BANISTER_DEFAULTS["tau_1"],
        tau_2: float = BANISTER_DEFAULTS["tau_2"],
        k_1: float = BANISTER_DEFAULTS["k_1"],
        k_2: float = BANISTER_DEFAULTS["k_2"],
        p_0: float = BANISTER_DEFAULTS["p_0"],
    ):
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.k_1 = k_1
        self.k_2 = k_2
        self.p_0 = p_0
        self._fitted = False
        self.fit_r_squared: Optional[float] = None

        self.alpha = 5.0
        self.beta = 0.5


    def fitness_at(
        self, t: float, session_days: np.ndarray, loads: np.ndarray
    ) -> float:
        """Accumulated fitness at time t from all sessions."""
        dt = t - session_days
        mask = dt >= 0
        return float(np.sum(loads[mask] * np.exp(-dt[mask] / self.tau_1)))

    def fatigue_at(
        self, t: float, session_days: np.ndarray, loads: np.ndarray
    ) -> float:
        """Accumulated fatigue at time t from all sessions."""
        dt = t - session_days
        mask = dt >= 0
        return float(np.sum(loads[mask] * np.exp(-dt[mask] / self.tau_2)))

    def performance_at(
        self, t: float, session_days: np.ndarray, loads: np.ndarray
    ) -> float:
        """Predicted performance at time t."""
        fit = self.fitness_at(t, session_days, loads)
        fat = self.fatigue_at(t, session_days, loads)
        return self.p_0 + self.k_1 * fit - self.k_2 * fat

    def performance_curve(
        self,
        session_days: np.ndarray,
        loads: np.ndarray,
        t_start: float = 0,
        t_end: Optional[float] = None,
        resolution: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute fitness, fatigue, and performance curves.

        Returns
        -------
        (time_points, fitness, fatigue, performance) — all np.ndarray
        """
        if t_end is None:
            t_end = float(session_days.max()) + 30

        t = np.arange(t_start, t_end + resolution, resolution)
        fitness = np.array([self.fitness_at(ti, session_days, loads) for ti in t])
        fatigue = np.array([self.fatigue_at(ti, session_days, loads) for ti in t])
        performance = self.p_0 + self.k_1 * fitness - self.k_2 * fatigue

        return t, fitness, fatigue, performance


    def fit_parameters(
        self,
        session_days: np.ndarray,
        loads: np.ndarray,
        assessment_days: np.ndarray,
        assessment_scores: np.ndarray,
        n_assessments_for_blending: Optional[int] = None,
    ) -> dict:
        """
        Fit model parameters using nonlinear least squares.

        Parameters
        ----------
        session_days : np.ndarray
            Day index of each training session.
        loads : np.ndarray
            Training load (TRIMP or tonnage) per session.
        assessment_days : np.ndarray
            Day index of each assessment.
        assessment_scores : np.ndarray
            Observed normalized performance scores.
        n_assessments_for_blending : int, optional
            If provided, blend fitted params with defaults (Section 10).

        Returns
        -------
        dict
            Fitted parameters and goodness of fit.
        """
        if len(assessment_scores) < 5:
            warnings.warn("< 5 assessments: parameters loosely constrained.")

        def model_predict(_, p0, k1, k2, tau2, delta):
            """Predict performance at each assessment day."""
            tau1 = tau2 + delta
            predicted = []
            for ad in assessment_days:
                dt_fit = ad - session_days
                mask_fit = dt_fit >= 0
                fit = np.sum(loads[mask_fit] * np.exp(-dt_fit[mask_fit] / tau1))
                fat = np.sum(loads[mask_fit] * np.exp(-dt_fit[mask_fit] / tau2))
                predicted.append(p0 + k1 * fit - k2 * fat)
            return np.array(predicted)

        tau1_min, tau1_max = BANISTER_BOUNDS["tau_1"]
        tau2_min, tau2_max = BANISTER_BOUNDS["tau_2"]
        delta_min = 1.0
        delta_max = tau1_max - tau2_min

        delta_init = max(delta_min, self.tau_1 - self.tau_2)
        x0 = [self.p_0, self.k_1, self.k_2, self.tau_2, delta_init]

        lower = [
            BANISTER_BOUNDS["p_0"][0], BANISTER_BOUNDS["k_1"][0],
            BANISTER_BOUNDS["k_2"][0], tau2_min, delta_min,
        ]
        upper = [
            BANISTER_BOUNDS["p_0"][1], BANISTER_BOUNDS["k_1"][1],
            BANISTER_BOUNDS["k_2"][1], tau2_max, delta_max,
        ]

        try:
            popt, pcov = optimize.curve_fit(
                model_predict,
                assessment_days,
                assessment_scores,
                p0=x0,
                bounds=(lower, upper),
                maxfev=10000,
                method="trf",
            )

            p0_fit, k1_fit, k2_fit, tau2_fit, delta_fit = popt
            tau1_fit = float(np.clip(tau2_fit + delta_fit, tau1_min, tau1_max))

            if n_assessments_for_blending is not None:
                N_threshold = 15
                w = min(1.0, n_assessments_for_blending / N_threshold)
                self.p_0 = w * p0_fit + (1 - w) * BANISTER_DEFAULTS["p_0"]
                self.k_1 = w * k1_fit + (1 - w) * BANISTER_DEFAULTS["k_1"]
                self.k_2 = w * k2_fit + (1 - w) * BANISTER_DEFAULTS["k_2"]
                self.tau_1 = w * tau1_fit + (1 - w) * BANISTER_DEFAULTS["tau_1"]
                self.tau_2 = w * tau2_fit + (1 - w) * BANISTER_DEFAULTS["tau_2"]
            else:
                self.p_0 = p0_fit
                self.k_1 = k1_fit
                self.k_2 = k2_fit
                self.tau_1 = tau1_fit
                self.tau_2 = tau2_fit

            predicted = model_predict(None, *popt)
            ss_res = np.sum((assessment_scores - predicted)**2)
            ss_tot = np.sum((assessment_scores - np.mean(assessment_scores))**2)
            r_squared = 1 - ss_res / (ss_tot + 1e-10)
            self.fit_r_squared = round(r_squared, 4)
            self._fitted = True

            return {
                "p_0": round(self.p_0, 3),
                "k_1": round(self.k_1, 4),
                "k_2": round(self.k_2, 4),
                "tau_1": round(self.tau_1, 2),
                "tau_2": round(self.tau_2, 2),
                "r_squared": self.fit_r_squared,
                "converged": True,
            }

        except (RuntimeError, ValueError) as e:
            self._fitted = False
            return {"converged": False, "error": str(e)}


    def session_contributions(
        self,
        session_days: np.ndarray,
        loads: np.ndarray,
        t_eval: np.ndarray,
    ) -> list[dict]:
        """
        Decompose total fitness/fatigue into per-session contributions.

        Returns list of dicts, one per session, each containing arrays
        of that session's decaying fitness and fatigue over t_eval.
        """
        contributions = []
        for i, (sd, w) in enumerate(zip(session_days, loads)):
            dt = t_eval - sd
            mask = dt >= 0
            fit_contrib = np.zeros_like(t_eval, dtype=float)
            fat_contrib = np.zeros_like(t_eval, dtype=float)
            fit_contrib[mask] = w * np.exp(-dt[mask] / self.tau_1)
            fat_contrib[mask] = w * np.exp(-dt[mask] / self.tau_2)
            contributions.append({
                "session_day": sd,
                "load": w,
                "fitness_contribution": fit_contrib,
                "fatigue_contribution": fat_contrib,
                "performance_contribution": self.k_1 * fit_contrib - self.k_2 * fat_contrib,
            })
        return contributions


    def adaptation_quality(self, fatigue: float, max_fatigue: float = 1.0) -> float:
        """
        Sigmoid discount on adaptation based on accumulated fatigue.

        q(fatigue) = 1 / (1 + exp(alpha * (fatigue_norm - beta)))

        Returns
        -------
        float
            Quality factor in [0, 1]. 1.0 = full adaptation, 0 = wasted.
        """
        if max_fatigue <= 0:
            return 1.0
        fatigue_norm = fatigue / max_fatigue
        q = 1.0 / (1.0 + np.exp(self.alpha * (fatigue_norm - self.beta)))
        return float(round(q, 4))

    def overreaching_level(self, q: float) -> OverreachingLevel:
        """Section 9e — Classify overreaching from adaptation quality."""
        if q > 0.7:
            return OverreachingLevel.NORMAL
        elif q >= 0.4:
            return OverreachingLevel.FUNCTIONAL
        else:
            return OverreachingLevel.NON_FUNCTIONAL


    @staticmethod
    def compute_acwr(
        daily_loads: np.ndarray,
        acute_window: int = 7,
        chronic_window: int = 28,
    ) -> dict:
        """
        Compute ACWR using exponentially weighted moving averages (EWMA).

        Parameters
        ----------
        daily_loads : np.ndarray
            Daily training load over time (0 for rest days).
        acute_window : int
        chronic_window : int

        Returns
        -------
        dict
            {"acwr": float, "acute_load": float, "chronic_load": float,
             "zone": str}
        """
        if len(daily_loads) < chronic_window:
            return {"acwr": 1.0, "zone": "insufficient_data",
                    "acute_load": 0.0, "chronic_load": 0.0}

        lambda_acute = 2.0 / (acute_window + 1)
        lambda_chronic = 2.0 / (chronic_window + 1)

        ewma_acute = 0.0
        ewma_chronic = 0.0

        for load in daily_loads:
            ewma_acute = lambda_acute * load + (1 - lambda_acute) * ewma_acute
            ewma_chronic = lambda_chronic * load + (1 - lambda_chronic) * ewma_chronic

        if ewma_chronic < 1e-6:
            acwr = 0.0
            zone = "detraining"
        else:
            acwr = ewma_acute / ewma_chronic
            if acwr < 0.6:
                zone = "detraining"
            elif acwr <= 1.3:
                zone = "safe"
            elif acwr <= 1.5:
                zone = "caution"
            else:
                zone = "high_risk"

        return {
            "acwr": round(acwr, 3),
            "acute_load": round(ewma_acute, 2),
            "chronic_load": round(ewma_chronic, 2),
            "zone": zone,
        }


    def optimize_taper(
        self,
        session_days: np.ndarray,
        loads: np.ndarray,
        race_day: float,
        avg_load: float,
        last_hard_range: tuple[int, int] = (5, 14),
        volume_fractions: list[float] = None,
    ) -> list[dict]:
        """
        Grid search over taper configurations. Returns top 3 strategies.

        Parameters
        ----------
        session_days, loads : existing training history
        race_day : float
            Day index of race.
        avg_load : float
            Average session load for taper sessions.
        last_hard_range : tuple
            Range of days before race for last hard session.
        volume_fractions : list[float]
            Fraction of normal volume during taper.

        Returns
        -------
        list[dict]
            Top 3 taper strategies sorted by predicted race-day performance.
        """
        if volume_fractions is None:
            volume_fractions = [0.4, 0.5, 0.6, 0.7]

        taper_patterns = ["linear", "exponential", "step"]
        results = []

        for last_hard_offset in range(last_hard_range[0], last_hard_range[1] + 1):
            last_hard_day = race_day - last_hard_offset
            for vol_frac in volume_fractions:
                for pattern in taper_patterns:
                    taper_days = []
                    taper_loads = []
                    for d in range(int(last_hard_day) + 1, int(race_day)):
                        days_into_taper = d - last_hard_day
                        taper_length = race_day - last_hard_day

                        if pattern == "linear":
                            frac = vol_frac * (1 - days_into_taper / taper_length)
                        elif pattern == "exponential":
                            frac = vol_frac * np.exp(-2 * days_into_taper / taper_length)
                        else:
                            frac = vol_frac if days_into_taper < taper_length / 2 else vol_frac * 0.5

                        if d % 2 == 0:
                            taper_days.append(d)
                            taper_loads.append(avg_load * max(frac, 0.1))

                    all_days = np.concatenate([session_days, taper_days])
                    all_loads = np.concatenate([loads, taper_loads])

                    perf = self.performance_at(race_day, all_days, all_loads)
                    results.append({
                        "last_hard_day_offset": last_hard_offset,
                        "volume_fraction": vol_frac,
                        "pattern": pattern,
                        "predicted_performance": round(perf, 3),
                        "n_taper_sessions": len(taper_days),
                    })

        results.sort(key=lambda x: x["predicted_performance"], reverse=True)
        return results[:3]


    def simulate_session(
        self,
        session_days: np.ndarray,
        loads: np.ndarray,
        proposed_day: float,
        proposed_load: float,
        horizon_days: int = 30,
    ) -> dict:
        """
        Simulate adding a proposed session and return performance curves.

        Returns curves for: without proposed, with proposed, difference.
        """
        t_eval = np.arange(proposed_day - 5, proposed_day + horizon_days + 1, 1.0)

        _, _, _, perf_without = self.performance_curve(
            session_days, loads, t_eval[0], t_eval[-1]
        )

        fatigue_at_proposal = self.fatigue_at(proposed_day, session_days, loads)
        max_fat = np.max([self.fatigue_at(d, session_days, loads) for d in session_days]) if len(session_days) > 0 else 1.0
        q = self.adaptation_quality(fatigue_at_proposal, max_fat)

        effective_load = float(proposed_load) * q
        aug_days = np.append(session_days, proposed_day)
        aug_loads = np.append(loads, effective_load)
        _, fit_with, fat_with, perf_with = self.performance_curve(
            aug_days, aug_loads, t_eval[0], t_eval[-1]
        )

        best_idx = np.argmax(perf_with)

        return {
            "time_points": t_eval.tolist(),
            "performance_without": perf_without.tolist(),
            "performance_with": perf_with.tolist(),
            "fitness_with": fit_with.tolist(),
            "fatigue_with": fat_with.tolist(),
            "adaptation_quality": q,
            "effective_load": round(effective_load, 3),
            "overreaching": self.overreaching_level(q).value,
            "optimal_day_offset": int(t_eval[best_idx] - proposed_day),
            "peak_performance": round(float(perf_with[best_idx]), 3),
            "removing_session_better": bool(np.max(perf_without) > np.max(perf_with)),
        }


"""
====================================================================
SECTION 9g — PREDICTION CONFIDENCE (KNN + Gower Distance)
====================================================================
SimilarityScorer scores how confident the engine should be in a
prediction by checking how many similar past sessions exist. Distance
is Gower (mixed numeric/categorical), with the historical range
extended to include the query value so a proposal outside the
historical span doesn't yield a normalized distance > 1.
"""

class SimilarityScorer:
    """
    K-nearest neighbors with Gower distance for mixed feature types.
    Scores prediction confidence based on historical similarity.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.history: list[dict] = []

    def add_session(self, features: dict, outcome: float):
        """Record a historical session with its outcome."""
        self.history.append({"features": features, "outcome": outcome})

    def gower_distance(self, a: dict, b: dict) -> float:
        """
        Compute Gower distance between two feature dicts.

        Handles mixed types:
        - Numeric: |a_i - b_i| / range_i
        - Categorical: 0 if same, 1 if different
        """
        keys = set(a.keys()) & set(b.keys())
        if not keys:
            return 1.0

        distances = []
        for key in keys:
            va, vb = a[key], b[key]
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                all_vals = [
                    h["features"].get(key) for h in self.history
                    if isinstance(h["features"].get(key), (int, float))
                ]
                all_vals.extend([va, vb])
                val_range = max(all_vals) - min(all_vals)
                if val_range > 0:
                    distances.append(abs(va - vb) / val_range)
                else:
                    distances.append(0.0)
            else:
                distances.append(0.0 if va == vb else 1.0)

        return sum(distances) / len(distances) if distances else 1.0

    def predict_confidence(self, proposed_features: dict) -> dict:
        """
        Find K nearest neighbors and score prediction confidence.

        Returns
        -------
        dict
            {"confidence": ConfidenceLevel, "confidence_pct": float,
             "n_matches": int, "cv": float, "reasoning": str}
        """
        if len(self.history) < 2:
            return {
                "confidence": ConfidenceLevel.LOW,
                "confidence_pct": 10.0,
                "n_matches": 0,
                "cv": float("inf"),
                "reasoning": "Insufficient training history for similarity matching.",
            }

        distances = [
            (i, self.gower_distance(proposed_features, h["features"]))
            for i, h in enumerate(self.history)
        ]
        distances.sort(key=lambda x: x[1])

        k_actual = min(self.k, len(distances))
        neighbors = distances[:k_actual]
        outcomes = np.array([self.history[i]["outcome"] for i, _ in neighbors])

        mean_outcome = np.mean(outcomes)
        if abs(mean_outcome) < 1e-10:
            cv = float("inf")
        else:
            cv = float(np.std(outcomes, ddof=1) / abs(mean_outcome))

        if k_actual >= 5 and cv < 0.15:
            level = ConfidenceLevel.HIGH
            pct = 80.0 + (0.15 - cv) / 0.15 * 15.0
        elif k_actual >= 3 and cv < 0.30:
            level = ConfidenceLevel.MEDIUM
            pct = 50.0 + (0.30 - cv) / 0.30 * 30.0
        else:
            level = ConfidenceLevel.LOW
            pct = max(10.0, 50.0 * (1 - cv) if cv < 1 else 10.0)

        avg_dist = np.mean([d for _, d in neighbors])
        reasoning = (
            f"Found {k_actual} similar sessions (avg distance: {avg_dist:.2f}). "
            f"Outcome CV: {cv:.3f}. "
            f"{'Consistent' if cv < 0.15 else 'Variable' if cv < 0.3 else 'Highly variable'} outcomes."
        )

        return {
            "confidence": level,
            "confidence_pct": round(min(pct, 95.0), 1),
            "n_matches": k_actual,
            "cv": round(cv, 4),
            "reasoning": reasoning,
        }


"""
====================================================================
SECTION 10 — MODEL PERSONALIZATION PROGRESSION
====================================================================
PersonalizationTracker tracks how much data has accumulated for each
adaptive component (banister_params, prediction_slope, readiness_
weights, etc.) and returns a 0..1 trust weight that downstream code
uses to blend personalized estimates with population defaults. The
mapping is min(1, count/N_threshold) where the threshold reflects how
much data each component needs before its fit is reliable.
"""

class PersonalizationTracker:
    """
    Tracks blending weights across features and computes aggregate
    personalization percentage.

    Each feature personalizes at a different rate (different threshold).
    """

    FEATURE_THRESHOLDS = {
        "banister_params": 15,
        "readiness_weights": 30,
        "prediction_slope": 15,
        "correlation_insights": 30,
        "adaptation_quality": 20,
        "overreaching_thresholds": 15,
        "taper_optimization": 15,
        "prediction_confidence": 30,
        "interference_detection": 10,
        "goal_conflict": 12,
    }

    def __init__(self):
        self.data_counts: dict[str, int] = {k: 0 for k in self.FEATURE_THRESHOLDS}

    def update_count(self, feature: str, count: int):
        """Set the data point count for a feature."""
        if feature in self.data_counts:
            self.data_counts[feature] = count

    def get_weight(self, feature: str) -> float:
        """Get blending weight w for a feature (0 = pure default, 1 = fully personal)."""
        threshold = self.FEATURE_THRESHOLDS.get(feature, 15)
        count = self.data_counts.get(feature, 0)
        return min(1.0, count / threshold)

    def blend(self, feature: str, fitted_value: float, default_value: float) -> float:
        """Blend fitted and default values based on personalization weight."""
        w = self.get_weight(feature)
        return w * fitted_value + (1 - w) * default_value

    def overall_personalization_pct(self) -> float:
        """Average personalization % across all features."""
        weights = [self.get_weight(f) for f in self.FEATURE_THRESHOLDS]
        return round(np.mean(weights) * 100, 1)

    def stage(self) -> int:
        """
        Current personalization stage (1-4).
        Based on overall % and time/data maturity.
        """
        pct = self.overall_personalization_pct()
        if pct < 15:
            return 1
        elif pct < 60:
            return 2
        elif pct < 90:
            return 3
        else:
            return 4


"""
====================================================================
SECTION 11 — DAILY READINESS SCORE
====================================================================
ReadinessScore turns a dict of daily inputs (sleep, fatigue, soreness,
acwr, stress, days_since_hard) into a 0-100 composite. Default weights
come from sports science; once a user has 30+ assessments paired with
metric data, ReadinessScore can re-fit weights via ridge regression.
Inverted scales (e.g. fatigue: lower is better) are encoded in the
default normalization ranges.
"""

class ReadinessScore:
    """
    Composite 0-100 score from multiple inputs.

    Default weights from sports science, personalized via ridge regression
    after 30+ sessions with metric data and assessments.
    """

    def __init__(self, weights: Optional[dict[str, float]] = None):
        self.weights = weights or DEFAULT_READINESS_WEIGHTS.copy()

    def compute(
        self,
        inputs: dict[str, float],
        normalize_ranges: Optional[dict[str, tuple[float, float]]] = None,
    ) -> dict:
        """
        Compute readiness score.

        Parameters
        ----------
        inputs : dict[str, float]
            Raw input values keyed by component name.
            Keys should match self.weights.
        normalize_ranges : dict
            {component: (worst_value, best_value)} for normalization.
            If not provided, assumes inputs are already 0-1 (1 = best).

        Returns
        -------
        dict
            {"score": int, "band": ReadinessBand, "components": dict}
        """
        default_ranges = {
            "sleep": (4.0, 9.0),
            "fatigue": (100.0, 0.0),
            "days_since_hard": (0.0, 3.0),
            "acwr": (2.0, 1.0),
            "stress": (10.0, 1.0),
            "soreness": (10.0, 1.0),
        }
        ranges = normalize_ranges or default_ranges

        components = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for component, weight in self.weights.items():
            raw = inputs.get(component)
            if raw is None:
                continue

            if component in ranges:
                lo, hi = ranges[component]
                if abs(hi - lo) < 1e-10:
                    normalized = 0.5
                else:
                    normalized = (raw - lo) / (hi - lo)
                normalized = np.clip(normalized, 0, 1)
            else:
                normalized = np.clip(raw, 0, 1)

            components[component] = {
                "raw": raw,
                "normalized": round(normalized, 3),
                "weight": weight,
                "contribution": round(normalized * weight, 4),
            }
            weighted_sum += normalized * weight
            total_weight += weight

        if total_weight > 0:
            score = (weighted_sum / total_weight) * 100
        else:
            score = 50.0

        score = int(np.clip(round(score), 0, 100))

        if score >= 85:
            band = ReadinessBand.GREEN
        elif score >= 65:
            band = ReadinessBand.YELLOW
        elif score >= 40:
            band = ReadinessBand.ORANGE
        else:
            band = ReadinessBand.RED

        return {"score": score, "band": band, "components": components}

    def personalize_weights(
        self,
        pre_session_metrics: np.ndarray,
        performance_outcomes: np.ndarray,
        feature_names: list[str],
        alpha_range: np.ndarray = None,
    ) -> dict:
        """
        Section 11 — Personalize readiness weights via ridge regression.

        Parameters
        ----------
        pre_session_metrics : np.ndarray
            Shape (n_sessions, n_features). Pre-session metric values.
        performance_outcomes : np.ndarray
            Shape (n_sessions,). Performance outcome for each session.
        feature_names : list[str]
        alpha_range : np.ndarray
            Regularization strengths to try via CV.

        Returns
        -------
        dict
            {"weights": dict, "r_squared": float, "best_alpha": float}
        """
        from sklearn.linear_model import RidgeCV

        if alpha_range is None:
            alpha_range = np.logspace(-3, 3, 50)

        if len(performance_outcomes) < 30:
            return {
                "weights": self.weights,
                "note": "Need 30+ sessions for personalization",
            }

        ridge = RidgeCV(alphas=alpha_range, cv=5)
        ridge.fit(pre_session_metrics, performance_outcomes)

        coefs = np.abs(ridge.coef_)
        total = np.sum(coefs)
        if total > 0:
            normalized_coefs = coefs / total
        else:
            normalized_coefs = np.ones(len(feature_names)) / len(feature_names)

        new_weights = {
            name: round(float(w), 4)
            for name, w in zip(feature_names, normalized_coefs)
        }
        self.weights = new_weights

        r_squared = ridge.score(pre_session_metrics, performance_outcomes)

        return {
            "weights": new_weights,
            "r_squared": round(r_squared, 4),
            "best_alpha": round(float(ridge.alpha_), 4),
        }


"""
====================================================================
SECTION 12 — INJURY RISK PREDICTION
====================================================================
InjuryRiskModel exposes:
  - compute_monotony_strain: 7-day monotony (mean/std) and strain
    (mean × std), the classic Foster overuse indicators.
  - predict_injury_risk: logistic combination of ACWR, monotony,
    strain, fatigue, soreness, readiness_trend → P(injury within 7d).
    The threshold at 0.40 is consistent across the level label (>= 0.40
    means "high") and the block_hard_sessions flag (>= 0.40 also blocks).
"""

class InjuryRiskModel:
    """
    Population-based logistic regression for injury probability.

    Uses coefficients from Gabbett 2016 and Foster 1998.
    INTENTIONALLY does NOT personalize (see planning doc rationale).
    """

    @staticmethod
    def compute_monotony_strain(daily_loads_7d: np.ndarray) -> dict:
        """
        Foster's training monotony and strain.

        Monotony = mean / stdev (high = repetitive = illness risk)
        Strain = sum * monotony
        """
        if len(daily_loads_7d) < 7:
            return {"monotony": 0.0, "strain": 0.0}

        mean_load = np.mean(daily_loads_7d)
        std_load = np.std(daily_loads_7d, ddof=1)

        if std_load < 1e-6:
            monotony = 10.0
        else:
            monotony = mean_load / std_load

        strain = np.sum(daily_loads_7d) * monotony

        return {
            "monotony": round(monotony, 3),
            "strain": round(strain, 2),
        }

    @staticmethod
    def predict_injury_risk(
        acwr: float,
        monotony: float,
        strain: float,
        fatigue: float = 0.0,
        soreness: float = 0.0,
        readiness_trend: float = 0.0,
    ) -> dict:
        """
        Predict P(injury in next 7 days).

        Parameters
        ----------
        acwr : float
        monotony : float
        strain : float
        fatigue : float
            Current fatigue from Banister model (normalized).
        soreness : float
            Self-reported soreness (1-10 scale).
        readiness_trend : float
            Slope of readiness over last 3 days (negative = declining).

        Returns
        -------
        dict
            {"risk_probability", "risk_level", "contributing_factors",
             "block_hard_sessions": bool}
        """
        c = INJURY_RISK_COEFFICIENTS
        z = (
            c["intercept"]
            + c["acwr"] * acwr
            + c["monotony"] * monotony
            + c["strain"] * strain
            + c["fatigue"] * fatigue
            + c["soreness"] * soreness
            + c["readiness_trend"] * readiness_trend
        )

        probability = 1.0 / (1.0 + np.exp(-z))
        probability = round(float(probability), 4)

        contributions = {
            "acwr": round(c["acwr"] * acwr, 3),
            "monotony": round(c["monotony"] * monotony, 3),
            "strain": round(c["strain"] * strain, 3),
            "fatigue": round(c["fatigue"] * fatigue, 3),
            "soreness": round(c["soreness"] * soreness, 3),
            "readiness_trend": round(c["readiness_trend"] * readiness_trend, 3),
        }
        top_contributors = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )[:3]

        if probability < 0.15:
            level = "low"
        elif probability < 0.30:
            level = "moderate"
        elif probability < 0.40:
            level = "elevated"
        else:
            level = "high"

        return {
            "risk_probability": probability,
            "risk_pct": round(probability * 100, 1),
            "risk_level": level,
            "block_hard_sessions": probability >= 0.40,
            "contributing_factors": top_contributors,
        }


"""
====================================================================
SECTION 13 — CONCURRENT TRAINING INTERFERENCE DETECTION
====================================================================
InterferenceDetector compares running performance with vs without a
recent heavy lift. Test selection: defaults to Mann-Whitney U whenever
either group has fewer than 15 observations, since the Shapiro-Wilk
normality test has very low power at small n and would routinely fail
to reject normality on data that's actually non-normal — incorrectly
funnelling small-sample comparisons into Welch's t-test.
"""

class InterferenceDetector:
    """
    Detects whether heavy lifting before running (or vice versa)
    impairs performance for a specific user.

    Uses two-sample t-test (or Mann-Whitney U if non-normal).
    """

    @staticmethod
    def detect_interference(
        assessments_with_prior_lift: list[float],
        assessments_without_prior_lift: list[float],
        min_per_group: int = 5,
        alpha: float = 0.05,
    ) -> dict:
        """
        Test whether prior heavy lifting impairs running performance.

        Parameters
        ----------
        assessments_with_prior_lift : list[float]
            Running assessment scores with heavy lower body lift within 48hrs.
        assessments_without_prior_lift : list[float]
            Running assessment scores without prior heavy lift.
        min_per_group : int
        alpha : float

        Returns
        -------
        dict
            {"interference_detected": bool, "test_used": str,
             "statistic": float, "p_value": float, "effect_size": float,
             "mean_with": float, "mean_without": float}
        """
        a = np.array(assessments_with_prior_lift)
        b = np.array(assessments_without_prior_lift)

        if len(a) < min_per_group or len(b) < min_per_group:
            return {
                "interference_detected": None,
                "note": f"Need >= {min_per_group} in each group. "
                        f"Got {len(a)} with, {len(b)} without.",
            }

        normality_n_threshold = 15
        if len(a) < normality_n_threshold or len(b) < normality_n_threshold:
            both_normal = False
        else:
            _, p_norm_a = stats.shapiro(a)
            _, p_norm_b = stats.shapiro(b)
            both_normal = p_norm_a > 0.05 and p_norm_b > 0.05

        if both_normal:
            stat, p_value = stats.ttest_ind(a, b, equal_var=False)
            test_used = "Welch's t-test"
        else:
            stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
            test_used = "Mann-Whitney U"

        pooled_std = np.sqrt(
            ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
            / (len(a) + len(b) - 2)
        )
        cohens_d = (np.mean(b) - np.mean(a)) / pooled_std if pooled_std > 0 else 0.0

        interference = p_value < alpha and abs(cohens_d) > 0.3

        return {
            "interference_detected": interference,
            "test_used": test_used,
            "statistic": round(float(stat), 3),
            "p_value": round(float(p_value), 5),
            "effect_size_cohens_d": round(cohens_d, 3),
            "mean_with_lift": round(float(np.mean(a)), 3),
            "mean_without_lift": round(float(np.mean(b)), 3),
        }


"""
====================================================================
SECTION 14 — GOAL CONFLICT DETECTION
====================================================================
GoalConflictDetector flags pairs of goals whose training requirements
likely interfere (e.g. concurrent strength + endurance peaks) and
issues a recommendation to prioritize. Also exposes
partial_correlation as a generic helper for downstream analysis.
"""

class GoalConflictDetector:
    """
    Rule-based conflict matrix + partial correlation for personalization.
    """

    @staticmethod
    def check_conflict(goal_a: Goal, goal_b: Goal) -> dict:
        """Check conflict between two goals using the matrix."""
        key = (goal_a.goal_type, goal_b.goal_type)
        key_rev = (goal_b.goal_type, goal_a.goal_type)
        severity = CONFLICT_MATRIX.get(key, CONFLICT_MATRIX.get(key_rev, 0))

        labels = {0: "none", 1: "low", 2: "moderate", 3: "high"}
        return {
            "goal_a": goal_a.description,
            "goal_b": goal_b.description,
            "conflict_severity": severity,
            "conflict_label": labels.get(severity, "unknown"),
            "recommendation": (
                "Consider prioritizing one goal."
                if severity >= 2
                else "Manageable with proper scheduling."
                if severity == 1
                else "No significant conflict."
            ),
        }

    @staticmethod
    def partial_correlation(
        x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> dict:
        """
        Compute partial correlation r_xy.z (X-Y controlling for Z).

        Used to detect whether running volume hurts squat improvement
        independent of lifting volume.

        Returns
        -------
        dict
            {"partial_r": float, "p_value": float, "interference": bool}
        """
        n = len(x)
        if n < 12:
            return {"error": "Need 12+ weeks of dual-domain data"}

        r_xy, _ = stats.pearsonr(x, y)
        r_xz, _ = stats.pearsonr(x, z)
        r_yz, _ = stats.pearsonr(y, z)

        denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        if denom < 1e-10:
            return {"partial_r": 0.0, "p_value": 1.0, "interference": False}

        partial_r = (r_xy - r_xz * r_yz) / denom

        t_stat = partial_r * np.sqrt((n - 3) / (1 - partial_r**2 + 1e-10))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 3))

        return {
            "partial_r": round(partial_r, 4),
            "p_value": round(p_value, 5),
            "interference": partial_r < -0.3 and p_value < 0.05,
        }


"""
====================================================================
SECTION 15 — DATA QUALITY SCORING
====================================================================
DataQualityScorer flags suspect entries (modified-z outliers,
implausible values, missing data) and returns a composite 0-100
quality score so downstream consumers can decide whether a fit/
prediction is trustworthy.
"""

class DataQualityScorer:
    """
    Tracks logging consistency, detects outliers, and computes composite
    data quality score.
    """

    @staticmethod
    def modified_z_score(values: np.ndarray) -> np.ndarray:
        """
        Robust outlier detection using modified Z-scores.

        Uses median and MAD (median absolute deviation) instead of
        mean and standard deviation.

        Returns
        -------
        np.ndarray
            Modified Z-scores. |z| > 3.5 → likely outlier.
        """
        median = np.median(values)
        mad = np.median(np.abs(values - median))

        if mad < 1e-10:
            return np.zeros_like(values)

        return 0.6745 * (values - median) / mad

    @staticmethod
    def detect_outliers(
        values: np.ndarray, threshold: float = 3.5
    ) -> dict:
        """
        Flag outliers in a metric series.

        Returns
        -------
        dict
            {"outlier_indices": list[int], "outlier_values": list,
             "n_outliers": int, "outlier_rate": float}
        """
        z = DataQualityScorer.modified_z_score(values)
        outlier_mask = np.abs(z) > threshold
        indices = np.where(outlier_mask)[0].tolist()

        return {
            "outlier_indices": indices,
            "outlier_values": values[outlier_mask].tolist(),
            "n_outliers": len(indices),
            "outlier_rate": round(len(indices) / max(len(values), 1), 4),
        }

    @staticmethod
    def composite_score(
        logging_rate: float,
        metric_rate: float,
        completeness: float,
        outlier_rate: float,
    ) -> dict:
        """
        Compute composite data quality score (0-1).

        DQ = 0.4*logging + 0.3*metrics + 0.2*completeness + 0.1*(1 - outliers)

        Returns
        -------
        dict
            {"score": float, "grade": str, "suggestions": list[str]}
        """
        dq = (
            0.4 * logging_rate
            + 0.3 * metric_rate
            + 0.2 * completeness
            + 0.1 * (1 - outlier_rate)
        )
        dq = round(np.clip(dq, 0, 1), 3)

        suggestions = []
        if logging_rate < 0.7:
            suggestions.append("Logging workouts more consistently would improve predictions.")
        if metric_rate < 0.5:
            suggestions.append("Tracking daily metrics like sleep and stress helps the engine learn faster.")
        if completeness < 0.6:
            suggestions.append("Adding RPE and heart rate data makes load estimates more accurate.")
        if outlier_rate > 0.05:
            suggestions.append("Some entries look unusual — double-check recent logs.")

        grade = "A" if dq >= 0.85 else "B" if dq >= 0.7 else "C" if dq >= 0.5 else "D"

        return {
            "score": dq,
            "grade": grade,
            "confidence_penalty": 0.2 if dq < 0.5 else 0.0,
            "suggestions": suggestions,
        }


"""
====================================================================
SECTION 16 — RETROSPECTIVE BLOCK ANALYSIS
====================================================================
BlockAnalysis compares training mesocycles after the fact:
  - cohens_d: pooled-std effect size between two score sets.
  - compare_blocks: always uses Welch's independent-samples t-test;
    two consecutive blocks are *not* paired even when they happen to
    contain the same number of assessments (no per-assessment matching
    structure).
  - detect_changepoints: CUSUM-style boundary detection in daily loads.
  - block_summary: sorts assessments by date before computing
    last-minus-first delta, so an out-of-order insertion doesn't
    produce a meaningless change number.
"""

class BlockAnalysis:
    """
    Analyze training blocks (mesocycles) for effectiveness.

    Includes Cohen's d effect size, paired t-test, and
    BOCPD-inspired changepoint detection.
    """

    @staticmethod
    def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
        """
        Cohen's d effect size between two groups.

        d = (mean_b - mean_a) / s_pooled

        Interpretation: |d| < 0.2 negligible, 0.2-0.5 small,
                        0.5-0.8 medium, >= 0.8 large.
        """
        na, nb = len(group_a), len(group_b)
        if na < 2 or nb < 2:
            return 0.0

        mean_a, mean_b = np.mean(group_a), np.mean(group_b)
        var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)

        pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
        if pooled_std < 1e-10:
            return 0.0

        return round((mean_b - mean_a) / pooled_std, 4)

    @staticmethod
    def compare_blocks(
        block_a_scores: np.ndarray,
        block_b_scores: np.ndarray,
    ) -> dict:
        """
        Compare two training blocks using effect size and paired t-test.

        Returns
        -------
        dict
            {"cohens_d", "d_interpretation", "t_stat", "p_value",
             "significant": bool, "summary": str}
        """
        d = BlockAnalysis.cohens_d(block_a_scores, block_b_scores)

        if abs(d) < 0.2:
            interp = "negligible"
        elif abs(d) < 0.5:
            interp = "small"
        elif abs(d) < 0.8:
            interp = "medium"
        else:
            interp = "large"

        t_stat, p_value = stats.ttest_ind(
            block_a_scores, block_b_scores, equal_var=False
        )
        test = "independent"

        direction = "improved" if d > 0 else "declined" if d < 0 else "unchanged"

        return {
            "cohens_d": d,
            "d_interpretation": interp,
            "test_type": test,
            "t_stat": round(float(t_stat), 3),
            "p_value": round(float(p_value), 5),
            "significant": p_value < 0.05,
            "summary": f"Performance {direction} with {interp} effect (d={d:.2f}, p={p_value:.3f}).",
        }

    @staticmethod
    def detect_changepoints(
        daily_loads: np.ndarray, threshold: float = 2.0
    ) -> list[int]:
        """
        Simplified changepoint detection using CUSUM-like approach.

        Detects significant shifts in training load that suggest
        natural block boundaries.

        For production, use Bayesian Online Changepoint Detection
        (Adams & MacKay 2007).

        Parameters
        ----------
        daily_loads : np.ndarray
        threshold : float
            Sensitivity (lower = more changepoints detected).

        Returns
        -------
        list[int]
            Indices of detected changepoints.
        """
        if len(daily_loads) < 14:
            return []

        mean = np.mean(daily_loads)
        std = np.std(daily_loads, ddof=1)
        if std < 1e-10:
            return []

        standardized = (daily_loads - mean) / std
        cusum_pos = np.zeros(len(standardized))
        cusum_neg = np.zeros(len(standardized))
        changepoints = []

        for i in range(1, len(standardized)):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + standardized[i] - 0.5)
            cusum_neg[i] = min(0, cusum_neg[i - 1] + standardized[i] + 0.5)

            if cusum_pos[i] > threshold or cusum_neg[i] < -threshold:
                changepoints.append(i)
                cusum_pos[i] = 0
                cusum_neg[i] = 0

        return changepoints

    @staticmethod
    def block_summary(
        sessions: list[dict],
        assessments: list[Assessment],
        block_start: date,
        block_end: date,
    ) -> dict:
        """
        Generate summary statistics for a training block.

        Parameters
        ----------
        sessions : list[dict]
            Each with "date", "type", "load" keys at minimum.
        assessments : list[Assessment]
        block_start, block_end : date

        Returns
        -------
        dict
            Volume, intensity distribution, assessment change, etc.
        """
        block_sessions = [
            s for s in sessions
            if block_start <= s["date"] <= block_end
        ]
        block_assessments = [
            a for a in assessments
            if block_start <= a.assessment_date <= block_end
        ]

        loads = [s["load"] for s in block_sessions]
        types = [s.get("type", "unknown") for s in block_sessions]

        type_dist = {}
        for t in types:
            type_dist[t] = type_dist.get(t, 0) + 1

        scored_assessments = sorted(
            (a for a in block_assessments if a.normalized_score is not None),
            key=lambda a: a.assessment_date,
        )
        assessment_scores = [a.normalized_score for a in scored_assessments]
        assessment_change = None
        if len(assessment_scores) >= 2:
            assessment_change = round(assessment_scores[-1] - assessment_scores[0], 3)

        return {
            "block_start": block_start.isoformat(),
            "block_end": block_end.isoformat(),
            "n_sessions": len(block_sessions),
            "total_load": round(sum(loads), 1) if loads else 0,
            "avg_load": round(np.mean(loads), 1) if loads else 0,
            "load_std": round(np.std(loads, ddof=1), 1) if len(loads) > 1 else 0,
            "session_type_distribution": type_dist,
            "n_assessments": len(block_assessments),
            "assessment_change": assessment_change,
            "days": (block_end - block_start).days,
        }


"""
====================================================================
SECTION 17 — POST-SESSION FEEDBACK LOOP
====================================================================
FeedbackLoop tracks the running bias between predicted and reported
effort (residual = predicted_effort − actual_rpe, EWMA-smoothed) and
nudges k_2 (the Banister fatigue coefficient) once a sustained bias
appears.

Direction is the *opposite* of the bias sign: when bias > 0 the model
is over-predicting effort, so we want to *decrease* k_2 (smaller
fatigue penalty → higher predicted performance → lower predicted
effort). The earlier implementation increased k_2 in that case, which
amplified the error rather than correcting it.
"""

class FeedbackLoop:
    """
    Tracks prediction error bias and applies lightweight parameter
    corrections between full refits.

    Uses EWMA bias tracking + online parameter nudging.
    """

    def __init__(
        self,
        lambda_: float = 0.15,
        bias_threshold: float = 1.0,
        nudge_rate: float = 0.02,
        min_sessions_for_nudge: int = 10,
    ):
        self.lambda_ = lambda_
        self.bias_threshold = bias_threshold
        self.nudge_rate = nudge_rate
        self.min_sessions_for_nudge = min_sessions_for_nudge
        self.bias = 0.0
        self.consecutive_biased = 0
        self.residuals: list[float] = []
        self.nudges_applied: list[dict] = []

    def record_feedback(
        self, predicted_effort: float, actual_rpe: float
    ) -> dict:
        """
        Record post-session feedback and update bias tracker.

        Parameters
        ----------
        predicted_effort : float
            Model's predicted effort level (comparable to RPE scale).
        actual_rpe : float
            User's reported RPE (1-10).

        Returns
        -------
        dict
            {"residual", "current_bias", "bias_detected": bool,
             "nudge_recommended": bool, "nudge_param": str}
        """
        residual = predicted_effort - actual_rpe
        self.residuals.append(residual)

        self.bias = self.lambda_ * residual + (1 - self.lambda_) * self.bias

        if abs(self.bias) > self.bias_threshold:
            self.consecutive_biased += 1
        else:
            self.consecutive_biased = 0

        bias_detected = self.consecutive_biased >= self.min_sessions_for_nudge
        nudge_param = "k_2" if bias_detected else None

        return {
            "residual": round(residual, 3),
            "current_bias": round(self.bias, 4),
            "bias_detected": bias_detected,
            "consecutive_biased": self.consecutive_biased,
            "nudge_recommended": bias_detected,
            "nudge_param": nudge_param,
        }

    def apply_nudge(self, model: BanisterModel) -> dict:
        """
        Apply a small parameter correction to the Banister model.

        Returns
        -------
        dict
            {"param_nudged", "old_value", "new_value", "direction"}
        """
        if abs(self.bias) <= self.bias_threshold:
            return {"nudge_applied": False, "reason": "Bias within threshold"}

        direction = -np.sign(self.bias)
        old_k2 = model.k_2
        model.k_2 += self.nudge_rate * direction

        model.k_2 = np.clip(
            model.k_2, BANISTER_BOUNDS["k_2"][0], BANISTER_BOUNDS["k_2"][1]
        )

        nudge_record = {
            "param_nudged": "k_2",
            "old_value": round(old_k2, 4),
            "new_value": round(model.k_2, 4),
            "direction": "increased" if direction > 0 else "decreased",
            "bias_at_nudge": round(self.bias, 4),
        }
        self.nudges_applied.append(nudge_record)

        self.consecutive_biased = 0

        return nudge_record


"""
====================================================================
SECTION 18 — RACE / ASSESSMENT POST-MORTEM
====================================================================
PostMortem decomposes the surprise (assessment − predicted) into
contributions from each tracked metric using a linearized
Shapley-style approximation: deviation_z × correlation_coef ×
gaussian_lag_weight. Output is a ranked list of "what helped / what
hurt" suitable for narrative summaries.
"""

class PostMortem:
    """
    Automated analysis of what contributed to an assessment result.

    Uses linearized Shapley value approximation:
      contribution = deviation_from_norm * correlation_coefficient * lag_weight
    """

    @staticmethod
    def analyze(
        assessment_score: float,
        predicted_score: float,
        metric_values_lead_up: dict[str, np.ndarray],
        metric_norms: dict[str, tuple[float, float]],
        correlation_coefficients: dict[str, float],
        optimal_lags: dict[str, int],
        lead_up_days: int = 7,
    ) -> dict:
        """
        Generate post-mortem for an assessment.

        Parameters
        ----------
        assessment_score : float
            Actual normalized score.
        predicted_score : float
            Model-predicted score.
        metric_values_lead_up : dict
            {metric_name: array of values in the lead_up_days before assessment}
        metric_norms : dict
            {metric_name: (historical_mean, historical_std)}
        correlation_coefficients : dict
            {metric_name: lagged Pearson r from correlation engine}
        optimal_lags : dict
            {metric_name: optimal lag in days}
        lead_up_days : int

        Returns
        -------
        dict
            {"surprise": float, "direction": str,
             "top_positive_contributors": list,
             "top_negative_contributors": list,
             "summary": str}
        """
        surprise = assessment_score - predicted_score
        direction = "better" if surprise > 0 else "worse" if surprise < 0 else "as expected"

        contributions = []

        for metric_name, values in metric_values_lead_up.items():
            if metric_name not in metric_norms or metric_name not in correlation_coefficients:
                continue

            mean, std = metric_norms[metric_name]
            if std < 1e-10:
                continue

            lag = optimal_lags.get(metric_name, 0)
            r = correlation_coefficients[metric_name]

            if len(values) == 0:
                continue
            avg_lead_up = np.mean(values)
            deviation = (avg_lead_up - mean) / std

            lag_weight = np.exp(-0.5 * ((lead_up_days - lag) / max(lag, 1))**2)

            contribution = deviation * r * lag_weight

            contributions.append({
                "metric": metric_name,
                "contribution": round(contribution, 4),
                "deviation_z": round(deviation, 3),
                "avg_lead_up": round(avg_lead_up, 3),
                "norm_mean": round(mean, 3),
                "correlation": round(r, 3),
            })

        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        positive = [c for c in contributions if c["contribution"] > 0]
        negative = [c for c in contributions if c["contribution"] < 0]

        summary_parts = []
        if direction == "better":
            summary_parts.append(
                f"Performance was {abs(surprise):.1f} units above prediction."
            )
            if positive:
                top = positive[0]
                summary_parts.append(
                    f"Top contributor: {top['metric']} averaged {top['avg_lead_up']:.1f} "
                    f"vs norm of {top['norm_mean']:.1f}."
                )
        elif direction == "worse":
            summary_parts.append(
                f"Performance was {abs(surprise):.1f} units below prediction."
            )
            if negative:
                top = negative[0]
                summary_parts.append(
                    f"Biggest drag: {top['metric']} averaged {top['avg_lead_up']:.1f} "
                    f"vs norm of {top['norm_mean']:.1f}."
                )
        else:
            summary_parts.append("Performance matched prediction closely.")

        return {
            "surprise": round(surprise, 3),
            "direction": direction,
            "top_positive_contributors": positive[:3],
            "top_negative_contributors": negative[:3],
            "all_contributions": contributions,
            "summary": " ".join(summary_parts),
        }


"""
====================================================================
SECTION 8 — ADAPTIVE TRAINING PLAN (Rule-Based + Thompson Sampling)
====================================================================
AdaptivePlan picks the next session:
  - rule_based_suggestions: applies a coaching rulebook (safety
    overrides for injury risk / very low readiness, ACWR guardrails,
    hard/easy alternation, deload after sustained progressive
    overload). The deload check buckets sessions into 7-day windows
    *by date*, not by list index — slicing recent_sessions in chunks
    of 7 only worked when training cadence was exactly 1/day.
  - ThompsonBandit: Beta-Bernoulli multi-armed bandit that learns
    which session types are paying off for this user via posterior
    sampling.
"""

class AdaptivePlan:
    """
    Phase 1: Rule-based session suggestions from sports science.
    Phase 2: Thompson Sampling contextual bandit for session type selection.

    Rule-based guardrails always apply, even when the bandit is active.
    """


    @staticmethod
    def rule_based_suggestions(
        recent_sessions: list[dict],
        readiness_score: int,
        injury_risk: float,
        acwr: float,
        adaptation_quality: float,
        days_since_hard: int,
    ) -> list[dict]:
        """
        Generate ranked session suggestions from codified coaching rules.

        Parameters
        ----------
        recent_sessions : list[dict]
            Last 21 days of sessions with "date", "type", "load", "is_hard".
        readiness_score : int (0-100)
        injury_risk : float (0-1)
        acwr : float
        adaptation_quality : float (0-1)
        days_since_hard : int

        Returns
        -------
        list[dict]
            Ranked suggestions: [{"type", "reason", "priority"}, ...]
        """
        suggestions = []

        if injury_risk > 0.40:
            return [{"type": "rest", "reason": "Injury risk elevated (> 40%). Rest recommended.", "priority": 1}]

        if readiness_score < 40:
            return [{"type": "rest", "reason": f"Readiness very low ({readiness_score}). Rest day.", "priority": 1}]

        if recent_sessions:
            anchor = max(s["date"] for s in recent_sessions)
            weekly_loads = [0.0, 0.0, 0.0]
            weekly_counts = [0, 0, 0]
            for s in recent_sessions:
                days_back = (anchor - s["date"]).days
                if 0 <= days_back < 21:
                    week_idx = 2 - (days_back // 7)
                    weekly_loads[week_idx] += s.get("load", 0)
                    weekly_counts[week_idx] += 1
            if all(c > 0 for c in weekly_counts) and all(
                weekly_loads[i] < weekly_loads[i + 1]
                for i in range(len(weekly_loads) - 1)
            ):
                suggestions.append({
                    "type": "deload_week",
                    "reason": "3+ weeks progressive overload. Consider deload.",
                    "priority": 2,
                })

        if acwr > 1.5:
            suggestions.append({
                "type": "easy",
                "reason": f"ACWR high ({acwr:.2f}). Easy session only.",
                "priority": 1,
            })
            return suggestions

        if acwr < 0.6:
            suggestions.append({
                "type": "moderate",
                "reason": f"ACWR low ({acwr:.2f}). Risk of detraining — add volume gradually.",
                "priority": 2,
            })

        if days_since_hard < 2:
            suggestions.append({
                "type": "easy",
                "reason": f"Last hard session was {days_since_hard} day(s) ago. Recovery needed.",
                "priority": 2,
            })
        elif readiness_score >= 85 and days_since_hard >= 2:
            suggestions.append({
                "type": "hard",
                "reason": f"Well recovered (readiness {readiness_score}, {days_since_hard} days since hard). Good for quality session.",
                "priority": 3,
            })
        elif readiness_score >= 65:
            suggestions.append({
                "type": "moderate",
                "reason": f"Moderate readiness ({readiness_score}). Steady session appropriate.",
                "priority": 3,
            })
        else:
            suggestions.append({
                "type": "easy",
                "reason": f"Below-average readiness ({readiness_score}). Easy day.",
                "priority": 2,
            })

        if adaptation_quality < 0.4:
            suggestions.append({
                "type": "rest",
                "reason": f"Adaptation efficiency low ({adaptation_quality:.0%}). Additional training likely counterproductive.",
                "priority": 1,
            })

        suggestions.sort(key=lambda x: x["priority"])
        return suggestions


    class ThompsonBandit:
        """
        Multi-armed bandit for session type selection.

        Each arm is a session type. Reward = improvement at next assessment.
        Uses Beta distribution: Beta(successes + 1, failures + 1).
        """

        def __init__(self, arms: list[str]):
            self.arms = arms
            self.successes: dict[str, int] = {a: 0 for a in arms}
            self.failures: dict[str, int] = {a: 0 for a in arms}

        def sample_and_recommend(self) -> tuple[str, dict[str, float]]:
            """
            Sample from each arm's Beta distribution and recommend highest.

            Returns
            -------
            (recommended_arm, sampled_values)
            """
            samples = {}
            for arm in self.arms:
                alpha = self.successes[arm] + 1
                beta = self.failures[arm] + 1
                samples[arm] = float(np.random.beta(alpha, beta))

            recommended = max(samples, key=samples.get)
            return recommended, samples

        def update(self, arm: str, improved: bool):
            """Update arm's distribution based on outcome."""
            if arm not in self.arms:
                raise ValueError(f"Unknown arm: {arm}")
            if improved:
                self.successes[arm] += 1
            else:
                self.failures[arm] += 1

        def get_stats(self) -> dict:
            """Current state of all arms."""
            stats_out = {}
            for arm in self.arms:
                a = self.successes[arm] + 1
                b = self.failures[arm] + 1
                stats_out[arm] = {
                    "successes": self.successes[arm],
                    "failures": self.failures[arm],
                    "mean": round(a / (a + b), 4),
                    "total_trials": self.successes[arm] + self.failures[arm],
                }
            return stats_out


"""
====================================================================
MODEL DRIFT DETECTION (Section 10 supplement)
====================================================================
DriftDetector runs a CUSUM control chart on prediction residuals to
flag when the Banister fit no longer matches reality and a refit is
warranted. Holds off the CUSUM update during the first 4 residuals
(returns warming_up=True) so a single early outlier doesn't trip the
threshold before there's enough data to standardize residuals.
"""

class DriftDetector:
    """
    CUSUM control chart for detecting when the Banister model's
    predictions drift from reality, triggering a refit.
    """

    def __init__(self, threshold: float = 4.0, drift_slack: float = 0.5):
        self.threshold = threshold
        self.drift_slack = drift_slack
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.residuals: list[float] = []

    def add_residual(self, predicted: float, observed: float) -> dict:
        """
        Add a new prediction residual and check for drift.

        Returns
        -------
        dict
            {"residual", "cusum_pos", "cusum_neg",
             "drift_detected": bool, "refit_recommended": bool}
        """
        residual = predicted - observed
        self.residuals.append(residual)

        if len(self.residuals) < 5:
            return {
                "residual": round(residual, 3),
                "cusum_pos": round(self.cusum_pos, 3),
                "cusum_neg": round(self.cusum_neg, 3),
                "drift_detected": False,
                "refit_recommended": False,
                "warming_up": True,
            }

        std = np.std(self.residuals, ddof=1)
        z = residual / std if std > 0 else 0.0

        self.cusum_pos = max(0, self.cusum_pos + z - self.drift_slack)
        self.cusum_neg = min(0, self.cusum_neg + z + self.drift_slack)

        drift = self.cusum_pos > self.threshold or self.cusum_neg < -self.threshold

        if drift:
            self.cusum_pos = 0
            self.cusum_neg = 0

        return {
            "residual": round(residual, 3),
            "cusum_pos": round(self.cusum_pos, 3),
            "cusum_neg": round(self.cusum_neg, 3),
            "drift_detected": drift,
            "refit_recommended": drift,
        }


"""
====================================================================
PLATEAU DETECTION (Section 10 supplement)
====================================================================
PlateauDetector flags stalled progress in a normalized-score series:
combines a linear-fit slope test (is the trend statistically flat?)
with a CUSUM-style range check, then surfaces both the verdict and a
suggested mitigation (e.g. block periodization, deload, or move to a
new training stimulus).
"""

class PlateauDetector:
    """
    Detects performance plateaus by testing whether the slope of
    the last N assessments is significantly different from zero.
    """

    @staticmethod
    def detect_plateau(
        assessment_scores: np.ndarray,
        assessment_days: np.ndarray,
        min_assessments: int = 6,
        alpha: float = 0.05,
    ) -> dict:
        """
        Test whether recent assessments show a plateau (slope ≈ 0).

        Returns
        -------
        dict
            {"is_plateau": bool, "slope", "p_value", "r_squared", "message"}
        """
        n = len(assessment_scores)
        if n < min_assessments:
            return {"is_plateau": False, "note": f"Need {min_assessments}+ assessments"}

        recent_scores = assessment_scores[-min_assessments:]
        recent_days = assessment_days[-min_assessments:]

        slope, intercept, r_value, p_value, se = stats.linregress(
            recent_days, recent_scores
        )

        is_plateau = p_value > alpha
        is_declining = slope < 0 and p_value < alpha

        if is_plateau:
            message = (
                f"Performance has plateaued over the last {min_assessments} assessments "
                f"(slope = {slope:.4f}, p = {p_value:.3f}). Consider changing training stimulus."
            )
        elif is_declining:
            message = (
                f"Performance is declining (slope = {slope:.4f}, p = {p_value:.3f}). "
                f"Review recovery, stress, and training load."
            )
        else:
            message = f"Performance is improving (slope = {slope:.4f})."

        return {
            "is_plateau": is_plateau,
            "is_declining": is_declining,
            "slope": round(slope, 6),
            "p_value": round(p_value, 4),
            "r_squared": round(r_value**2, 4),
            "message": message,
        }



class FitnessOptimizer:
    """
    Top-level orchestrator that wires all subsystems together.

    Typical workflow:
      1. Log sessions (running, lifting) and daily metrics
      2. Record assessments periodically
      3. Engine computes readiness, injury risk, predictions
      4. Simulator allows what-if exploration
      5. Correlation engine discovers personal patterns
      6. Adaptive plan suggests next sessions
    """

    def __init__(self, resting_hr: Optional[int] = None):
        self.banister = BanisterModel()
        self.prediction_engine = PredictionEngine()
        self.correlation_engine = CorrelationEngine()
        self.readiness = ReadinessScore()
        self.similarity = SimilarityScorer()
        self.personalization = PersonalizationTracker()
        self.feedback = FeedbackLoop()
        self.drift_detector = DriftDetector()

        self.goals: list[Goal] = []
        self.running_sessions: list[RunningSession] = []
        self.lifting_sessions: list[LiftingSession] = []
        self.metric_definitions: list[CustomMetricDefinition] = []
        self.metric_entries: list[MetricEntry] = []
        self.assessments: list[Assessment] = []

        self.resting_hr: Optional[int] = resting_hr

        self.readiness_history: list[tuple[date, int]] = []

        self.bandit = AdaptivePlan.ThompsonBandit([
            "easy_run", "intervals", "tempo", "long_run", "hill_repeats",
            "heavy_lift", "moderate_lift", "rest",
        ])

    def _estimate_resting_hr(self) -> Optional[int]:
        """
        Estimate resting HR from logged easy-effort sessions.

        Uses the 5th-percentile avg_heart_rate across easy/Z2 sessions, then
        subtracts ~10 bpm (avg-during-exercise to true rest is a sizeable
        gap, but easy avg HR is a much better individualized anchor than 60
        for HR-based TRIMP). Returns None if there's not enough data.
        """
        easy_types = {RunWorkoutType.EASY, RunWorkoutType.LONG_RUN}
        easy_hrs = [
            s.avg_heart_rate for s in self.running_sessions
            if s.workout_type in easy_types and s.avg_heart_rate
        ]
        if len(easy_hrs) < 5:
            return None
        floor = float(np.percentile(easy_hrs, 5))
        return max(40, int(round(floor - 10)))


    def add_running_session(self, session: RunningSession) -> float:
        """Add running session, compute TRIMP, store."""
        if self.resting_hr is None:
            self.resting_hr = self._estimate_resting_hr()
        trimp = TrainingLoad.compute_trimp(session, hr_rest=self.resting_hr)
        self.running_sessions.append(session)
        return trimp

    def add_lifting_session(self, session: LiftingSession) -> float:
        """Add lifting session, compute tonnage, store."""
        tonnage = TrainingLoad.compute_tonnage(session)
        self.lifting_sessions.append(session)
        return tonnage

    def add_assessment(self, assessment: Assessment) -> Assessment:
        """Add assessment, normalize, update models."""
        assessment = AssessmentNormalizer.normalize_assessment(
            assessment, self.assessments
        )
        self.assessments.append(assessment)

        n = len(self.assessments)
        self.personalization.update_count("banister_params", n)

        if n >= PredictionEngine.MIN_ASSESSMENTS:
            try:
                self.prediction_engine.fit(self.assessments)
                self.personalization.update_count("prediction_slope", n)
            except (ValueError, RuntimeError):
                pass

        return assessment


    def daily_readiness(
        self, inputs: dict[str, float], today: Optional[date] = None
    ) -> dict:
        """
        Compute today's readiness score and append it to readiness_history
        so recent_readiness_trend can fit a slope across recent days.

        Parameters
        ----------
        inputs : dict
            Same keys ReadinessScore.compute expects.
        today : date, optional
            Date to associate with this score. Defaults to date.today().
            Pass an explicit date when backfilling historical readiness.
        """
        result = self.readiness.compute(inputs)
        score = result.get("score")
        if score is not None:
            stamp = today if today is not None else date.today()
            self.readiness_history = [
                (d, s) for (d, s) in self.readiness_history if d != stamp
            ]
            self.readiness_history.append((stamp, int(score)))
            self.readiness_history.sort(key=lambda t: t[0])
        return result

    def recent_readiness_trend(self, days: int = 3) -> float:
        """
        Slope of the most recent `days` readiness scores (points/day).

        Negative = readiness declining, positive = improving. Returns 0.0
        when there's not enough history (need >= 2 entries within the window).
        """
        if len(self.readiness_history) < 2 or days < 2:
            return 0.0
        recent = self.readiness_history[-days:]
        if len(recent) < 2:
            return 0.0
        anchor = recent[0][0]
        xs = np.array([(d - anchor).days for d, _ in recent], dtype=float)
        ys = np.array([s for _, s in recent], dtype=float)
        if np.std(xs) < 1e-10:
            return 0.0
        slope, _, _, _, _ = stats.linregress(xs, ys)
        return float(slope)

    def daily_injury_risk(
        self,
        daily_loads_7d: np.ndarray,
        acwr: float,
        fatigue: float = 0.0,
        soreness: float = 0.0,
        readiness_trend: float = 0.0,
    ) -> dict:
        """Compute today's injury risk indicators."""
        ms = InjuryRiskModel.compute_monotony_strain(daily_loads_7d)
        return InjuryRiskModel.predict_injury_risk(
            acwr=acwr,
            monotony=ms["monotony"],
            strain=ms["strain"],
            fatigue=fatigue,
            soreness=soreness,
            readiness_trend=readiness_trend,
        )

    def suggest_sessions(
        self,
        recent_sessions: list[dict],
        readiness_score: int,
        injury_risk: float,
        acwr: float,
        adaptation_quality: float,
        days_since_hard: int,
    ) -> list[dict]:
        """Get ranked session suggestions."""
        return AdaptivePlan.rule_based_suggestions(
            recent_sessions, readiness_score, injury_risk,
            acwr, adaptation_quality, days_since_hard,
        )


    def status(self) -> dict:
        """Current system status and personalization level."""
        return {
            "n_running_sessions": len(self.running_sessions),
            "n_lifting_sessions": len(self.lifting_sessions),
            "n_assessments": len(self.assessments),
            "n_metric_entries": len(self.metric_entries),
            "n_goals": len(self.goals),
            "personalization_pct": self.personalization.overall_personalization_pct(),
            "personalization_stage": self.personalization.stage(),
            "banister_fitted": self.banister._fitted,
            "banister_r_squared": self.banister.fit_r_squared,
            "prediction_fitted": self.prediction_engine._fitted,
        }



try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    _HAS_GUI = True
except ImportError:
    _HAS_GUI = False


"""
====================================================================
DEMO DATA GENERATOR
====================================================================
generate_demo_data populates a fresh FitnessOptimizer with six months
of synthetic but realistic training: ~80 running sessions across
workout types, ~50 lifting sessions, 12 spaced 5K assessments, and
840 daily metric entries (sleep / stress / soreness / mood / energy).
Used by the GUI on startup so every panel has data to render before
the user logs anything; also used by run_smoke_tests for CI-style
validation.
"""

def generate_demo_data(optimizer: FitnessOptimizer):
    """Populate the optimizer with 6 months of realistic synthetic data."""
    rng = np.random.RandomState(42)
    base = date(2025, 1, 1)

    g1 = Goal("g1", GoalType.ENDURANCE, "Marathon < 3:00:00", 180.0, 210.0,
              date(2026, 10, 1), base)
    g2 = Goal("g2", GoalType.STRENGTH, "Squat 250 lbs", 250.0, 200.0,
              date(2027, 1, 1), base)
    GoalEngine.generate_milestones(g1, n_milestones=4)
    GoalEngine.generate_milestones(g2, n_milestones=4)
    optimizer.goals = [g1, g2]

    run_types = [RunWorkoutType.EASY, RunWorkoutType.TEMPO,
                 RunWorkoutType.INTERVALS, RunWorkoutType.LONG_RUN]
    for week in range(24):
        n_runs = rng.choice([3, 4])
        run_days = sorted(rng.choice(7, n_runs, replace=False))
        for rd in run_days:
            d = base + timedelta(days=week * 7 + int(rd))
            wtype = rng.choice(run_types)
            dur = {"EASY": 40, "TEMPO": 35, "INTERVALS": 45,
                   "LONG_RUN": 75}.get(wtype.name, 45) + rng.randint(-5, 10)
            hr = {"EASY": 140, "TEMPO": 165, "INTERVALS": 170,
                  "LONG_RUN": 148}.get(wtype.name, 150) + rng.randint(-5, 5)
            rpe = {"EASY": 4, "TEMPO": 7, "INTERVALS": 8,
                   "LONG_RUN": 6}.get(wtype.name, 5) + rng.uniform(-0.5, 0.5)
            dist = dur * rng.uniform(0.14, 0.2)
            sess = RunningSession(
                session_id=f"run_{week}_{rd}", session_date=d,
                workout_type=wtype, duration_minutes=float(dur),
                distance_km=round(dist, 2), avg_heart_rate=int(hr),
                max_heart_rate=int(hr + 20), rpe=round(rpe, 1),
            )
            optimizer.add_running_session(sess)

    for week in range(24):
        for lift_day_offset in [1, 4]:
            d = base + timedelta(days=week * 7 + lift_day_offset)
            weight_base = 135 + week * 1.5
            exercises = [
                LiftingExercise("Squat", ["quads", "glutes"], 4,
                                [8, 8, 8, 8],
                                round(weight_base + rng.uniform(-10, 10), 1)),
                LiftingExercise("Bench Press", ["chest", "triceps"], 3,
                                [10, 10, 10],
                                round(weight_base * 0.7 + rng.uniform(-5, 5), 1)),
                LiftingExercise("Deadlift", ["back", "hamstrings"], 3,
                                [5, 5, 5],
                                round(weight_base * 1.2 + rng.uniform(-10, 10), 1)),
            ]
            sess = LiftingSession(
                session_id=f"lift_{week}_{lift_day_offset}", session_date=d,
                exercises=exercises,
                session_rpe=round(rng.uniform(6, 9), 1),
            )
            optimizer.add_lifting_session(sess)

    for i in range(12):
        d = base + timedelta(days=14 * i + 13)
        vdot = 45 + i * 0.4 + rng.uniform(-0.5, 0.5)
        a = Assessment(
            assessment_id=f"assess_{i}", assessment_date=d,
            domain="running",
            raw_value=round(1200 - i * 8 + rng.uniform(-5, 5), 1),
            raw_unit="seconds", assessment_type="5K",
            normalized_score=round(vdot, 2),
        )
        optimizer.add_assessment(a)

    for day_offset in range(168):
        d = base + timedelta(days=day_offset)
        for mid, val_fn in [
            ("sleep",    lambda: round(rng.uniform(5.5, 9.0), 1)),
            ("stress",   lambda: round(rng.uniform(1, 8), 1)),
            ("soreness", lambda: round(rng.uniform(1, 7), 1)),
            ("mood",     lambda: round(rng.uniform(3, 10), 1)),
            ("energy",   lambda: round(rng.uniform(3, 9), 1)),
        ]:
            optimizer.metric_entries.append(
                MetricEntry(metric_id=mid, entry_date=d, value=val_fn())
            )

    return optimizer


"""
====================================================================
MAIN GUI CLASS — FitnessOptimizerGUI (Tkinter)
====================================================================
14-tab desktop dashboard wrapping the full analytics surface:
Dashboard, Goals, Running/Lifting Logs, Assessments, Predictions,
Correlations, Simulator, Readiness, Injury Risk, Adaptive Plan, Block
Analysis, Data Quality, and Log Data.

Architecture: builders are idempotent — `_build_X_panel(parent)`
populates a tab from `self.optimizer` state plus precomputed caches
on `self`. `refresh()` is the single entry point for "data changed,
re-render everything": it calls `_precompute()` then destroys and
rebuilds each tab's children. The Log Data tab's form-submit handlers
ingest into `self.optimizer` and call `refresh()`, so adding a session
or assessment instantly propagates to every other panel. Per-panel
exception handling in refresh() keeps a single broken builder from
taking the GUI down — the failing panel surfaces an inline error
label instead.
"""

class FitnessOptimizerGUI:
    """Full Tkinter desktop application — 13 tabs covering all 18 sections."""

    BG      = "#1a1a2e"
    BG2     = "#16213e"
    BG3     = "#0f3460"
    FG      = "#e0e0e0"
    ACCENT  = "#00b4d8"
    GREEN   = "#2ecc71"
    YELLOW  = "#f1c40f"
    ORANGE  = "#e67e22"
    RED     = "#e74c3c"
    CARD_BG = "#222244"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Personalized Fitness Optimizer v2")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 700)
        self.root.configure(bg=self.BG)

        self.optimizer = generate_demo_data(FitnessOptimizer())
        self._precompute()
        self._setup_styles()
        self._build_ui()

    def _precompute(self):
        opt = self.optimizer
        base = date(2025, 1, 1)
        n_days = 168

        self.daily_loads = np.zeros(n_days)
        for s in opt.running_sessions:
            idx = (s.session_date - base).days
            if 0 <= idx < n_days:
                self.daily_loads[idx] += s.trimp or 0
        for s in opt.lifting_sessions:
            idx = (s.session_date - base).days
            if 0 <= idx < n_days:
                self.daily_loads[idx] += (s.tonnage or 0) / 100

        self.acwr_result = BanisterModel.compute_acwr(self.daily_loads)

        self.session_days = np.array([
            (s.session_date - base).days for s in opt.running_sessions
        ], dtype=float)
        self.session_loads = np.array([
            s.trimp or 0 for s in opt.running_sessions
        ], dtype=float)

        self.assess_days = np.array([
            (a.assessment_date - base).days for a in opt.assessments
        ], dtype=float)
        self.assess_scores = np.array([
            a.normalized_score or 0 for a in opt.assessments
        ], dtype=float)

        self.metric_arrays = {}
        for mid in ("sleep", "stress", "soreness", "mood", "energy"):
            vals = [e.value for e in opt.metric_entries if e.metric_id == mid]
            self.metric_arrays[mid] = np.array(vals) if vals else np.array([])

        if len(self.session_days) > 0:
            self.max_fatigue_raw = max(
                opt.banister.fatigue_at(d, self.session_days, self.session_loads)
                for d in self.session_days
            )
            raw_fat = opt.banister.fatigue_at(
                float(n_days), self.session_days, self.session_loads
            )
            self.current_fatigue = (
                float(raw_fat / self.max_fatigue_raw)
                if self.max_fatigue_raw > 0 else 0.0
            )
        else:
            self.max_fatigue_raw = 1.0
            self.current_fatigue = 0.0

        sore = self.metric_arrays.get("soreness", np.array([]))
        self.current_soreness = float(sore[-1]) if len(sore) > 0 else 3.0

        hard_types = {
            RunWorkoutType.INTERVALS, RunWorkoutType.TEMPO,
            RunWorkoutType.HILL_REPEATS, RunWorkoutType.RACE,
            RunWorkoutType.TIME_TRIAL,
        }
        metric_by_date: dict[str, dict[date, float]] = {}
        for entry in opt.metric_entries:
            metric_by_date.setdefault(entry.metric_id, {})[entry.entry_date] = entry.value

        def _metric_for(mid: str, d: date) -> Optional[float]:
            val = metric_by_date.get(mid, {}).get(d)
            if val is not None:
                return float(val)
            arr = self.metric_arrays.get(mid, np.array([]))
            return float(np.mean(arr)) if len(arr) > 0 else None

        today_idx = n_days - 1
        for offset in range(6, -1, -1):
            day_idx = today_idx - offset
            if day_idx < 0:
                continue
            d = base + timedelta(days=day_idx)

            if len(self.session_days) > 0 and self.max_fatigue_raw > 0:
                raw = opt.banister.fatigue_at(
                    float(day_idx), self.session_days, self.session_loads
                )
                fatigue_input = float(min(100.0, raw / self.max_fatigue_raw * 100.0))
            else:
                fatigue_input = 30.0

            if day_idx >= 27:
                acwr_d = float(BanisterModel.compute_acwr(
                    self.daily_loads[:day_idx + 1])["acwr"])
            else:
                acwr_d = 1.0

            last_hard = max(
                (s.session_date for s in opt.running_sessions
                 if s.session_date <= d and s.workout_type in hard_types),
                default=None,
            )
            dsh = float((d - last_hard).days) if last_hard else 7.0

            inputs = {
                "sleep": _metric_for("sleep", d),
                "stress": _metric_for("stress", d),
                "soreness": _metric_for("soreness", d),
                "fatigue": fatigue_input,
                "acwr": acwr_d,
                "days_since_hard": dsh,
            }
            inputs = {k: v for k, v in inputs.items() if v is not None}
            opt.daily_readiness(inputs, today=d)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=self.BG, foreground=self.FG, borderwidth=0)
        style.configure("TNotebook", background=self.BG)
        style.configure("TNotebook.Tab", background=self.BG2, foreground=self.FG,
                         padding=[14, 6], font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", self.BG3)],
                  foreground=[("selected", self.ACCENT)])
        style.configure("TFrame", background=self.BG)
        style.configure("TLabel", background=self.BG, foreground=self.FG,
                         font=("Segoe UI", 10))
        style.configure("TButton", background=self.BG3, foreground=self.FG,
                         font=("Segoe UI", 10, "bold"), padding=6)
        style.map("TButton", background=[("active", self.ACCENT)])
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"),
                         foreground=self.ACCENT)
        style.configure("SubHeader.TLabel", font=("Segoe UI", 11, "bold"),
                         foreground=self.FG)
        style.configure("Card.TFrame", background=self.CARD_BG)
        style.configure("Card.TLabel", background=self.CARD_BG, foreground=self.FG)
        style.configure("Big.TLabel", font=("Segoe UI", 28, "bold"),
                         foreground=self.FG, background=self.CARD_BG)
        style.configure("Green.TLabel", foreground=self.GREEN,
                         background=self.CARD_BG, font=("Segoe UI", 12, "bold"))
        style.configure("Yellow.TLabel", foreground=self.YELLOW,
                         background=self.CARD_BG, font=("Segoe UI", 12, "bold"))
        style.configure("Red.TLabel", foreground=self.RED,
                         background=self.CARD_BG, font=("Segoe UI", 12, "bold"))
        style.configure("Treeview", background=self.BG2, foreground=self.FG,
                         fieldbackground=self.BG2, font=("Segoe UI", 9),
                         rowheight=24)
        style.configure("Treeview.Heading", background=self.BG3,
                         foreground=self.ACCENT, font=("Segoe UI", 9, "bold"))
        style.configure("TLabelframe", background=self.BG, foreground=self.ACCENT)
        style.configure("TLabelframe.Label", background=self.BG,
                         foreground=self.ACCENT, font=("Segoe UI", 10, "bold"))
        style.configure("TScale", background=self.BG)
        style.configure("TEntry", fieldbackground=self.BG2, foreground=self.FG)
        style.configure("TSpinbox", fieldbackground=self.BG2, foreground=self.FG)

    def _build_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)
        self._tab_specs = [
            ("Dashboard",      self._build_dashboard),
            ("Goals",          self._build_goals),
            ("Running Log",    self._build_running_log),
            ("Lifting Log",    self._build_lifting_log),
            ("Assessments",    self._build_assessments),
            ("Predictions",    self._build_predictions),
            ("Correlations",   self._build_correlations),
            ("Simulator",      self._build_simulator),
            ("Readiness",      self._build_readiness),
            ("Injury Risk",    self._build_injury_risk),
            ("Adaptive Plan",  self._build_adaptive_plan),
            ("Block Analysis", self._build_block_analysis),
            ("Data Quality",   self._build_data_quality),
            ("Log Data",       self._build_log_data),
        ]
        self._tab_frames: list[ttk.Frame] = []
        for name, builder in self._tab_specs:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=f"  {name}  ")
            self._tab_frames.append(frame)
            builder(frame)

    def refresh(self):
        """
        Re-derive cached state and rebuild every tab.

        Used after data ingestion (Log Data forms) so all panels — dashboard,
        predictions, correlations, etc. — reflect the new optimizer state.
        Tkinter doesn't support partial-tree updates ergonomically, so we
        destroy each tab's children and re-run its builder. The notebook
        widget itself is preserved so the active tab and tab order survive.
        """
        try:
            current_tab = self.notebook.index("current")
        except tk.TclError:
            current_tab = 0

        self._precompute()

        for frame, (_, builder) in zip(self._tab_frames, self._tab_specs):
            for child in frame.winfo_children():
                child.destroy()
            try:
                builder(frame)
            except Exception as exc:
                ttk.Label(
                    frame,
                    text=f"Error rebuilding panel: {exc}",
                    style="Red.TLabel",
                ).pack(padx=12, pady=12, anchor="w")

        try:
            self.notebook.select(current_tab)
        except tk.TclError:
            pass

    def _card(self, parent, row, col, title, value, subtitle="",
              color_style="Card.TLabel", colspan=1):
        frame = ttk.Frame(parent, style="Card.TFrame", padding=12)
        frame.grid(row=row, column=col, columnspan=colspan,
                   padx=6, pady=6, sticky="nsew")
        ttk.Label(frame, text=title, style="Card.TLabel",
                  font=("Segoe UI", 9)).pack(anchor="w")
        ttk.Label(frame, text=str(value),
                  style="Big.TLabel").pack(anchor="w", pady=(2, 0))
        if subtitle:
            ttk.Label(frame, text=subtitle, style=color_style,
                      font=("Segoe UI", 10)).pack(anchor="w")
        return frame

    def _embed_figure(self, parent, fig, row=0, col=0, colspan=1,
                      toolbar=False):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        widget = canvas.get_tk_widget()
        widget.grid(row=row, column=col, columnspan=colspan,
                    padx=4, pady=4, sticky="nsew")
        if toolbar:
            tb_frame = ttk.Frame(parent)
            tb_frame.grid(row=row + 1, column=col, columnspan=colspan,
                          sticky="ew")
            NavigationToolbar2Tk(canvas, tb_frame)
        canvas.draw()
        return canvas

    def _make_fig(self, w=6, h=3.5):
        fig = Figure(figsize=(w, h), dpi=100, facecolor=self.BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.BG2)
        ax.tick_params(colors=self.FG, labelsize=8)
        ax.xaxis.label.set_color(self.FG)
        ax.yaxis.label.set_color(self.FG)
        ax.title.set_color(self.ACCENT)
        for spine in ax.spines.values():
            spine.set_color("#444")
        return fig, ax

    def _text_panel(self, parent, text, row=0, col=0, h=12, w=80,
                    colspan=1):
        st = scrolledtext.ScrolledText(
            parent, wrap="word", height=h, width=w,
            bg=self.BG2, fg=self.FG, font=("Consolas", 9),
            insertbackground=self.FG, borderwidth=0,
        )
        st.grid(row=row, column=col, columnspan=colspan,
                padx=6, pady=4, sticky="nsew")
        st.insert("1.0", text)
        st.configure(state="disabled")
        return st

    def _build_dashboard(self, parent):
        parent.columnconfigure((0, 1, 2, 3), weight=1)
        parent.rowconfigure(2, weight=1)

        status = self.optimizer.status()

        readiness = self.optimizer.daily_readiness({
            "sleep": 7.5, "fatigue": 30, "days_since_hard": 2,
            "acwr": self.acwr_result["acwr"], "stress": 4, "soreness": 3,
        })
        r_color = {"GREEN": "Green.TLabel", "YELLOW": "Yellow.TLabel",
                    "ORANGE": "Red.TLabel", "RED": "Red.TLabel"}.get(
            readiness["band"].name, "Card.TLabel")
        self._card(parent, 0, 0, "READINESS", readiness["score"],
                   readiness["band"].value, r_color)

        ms = InjuryRiskModel.compute_monotony_strain(self.daily_loads[-7:])
        risk = InjuryRiskModel.predict_injury_risk(
            self.acwr_result["acwr"], ms["monotony"], ms["strain"],
            fatigue=self.current_fatigue,
            soreness=self.current_soreness,
            readiness_trend=self.optimizer.recent_readiness_trend(days=3),
        )
        risk_color = ("Green.TLabel" if risk["risk_pct"] < 20 else
                      "Yellow.TLabel" if risk["risk_pct"] < 35 else "Red.TLabel")
        self._card(parent, 0, 1, "INJURY RISK", f"{risk['risk_pct']}%",
                   risk["risk_level"].upper(), risk_color)

        acwr_color = ("Green.TLabel" if self.acwr_result["zone"] == "safe"
                      else "Yellow.TLabel")
        self._card(parent, 0, 2, "ACWR", self.acwr_result["acwr"],
                   self.acwr_result["zone"].upper(), acwr_color)
        self._card(parent, 0, 3, "PERSONALIZATION",
                   f"{status['personalization_pct']}%",
                   f"Stage {status['personalization_stage']}")

        self._card(parent, 1, 0, "RUNNING SESSIONS",
                   status["n_running_sessions"])
        self._card(parent, 1, 1, "LIFTING SESSIONS",
                   status["n_lifting_sessions"])
        self._card(parent, 1, 2, "ASSESSMENTS", status["n_assessments"])
        self._card(parent, 1, 3, "METRIC ENTRIES",
                   status["n_metric_entries"])

        fig, ax = self._make_fig(12, 3.5)
        weeks, weekly_run, weekly_lift = [], [], []
        base = date(2025, 1, 1)
        for w in range(24):
            start = base + timedelta(weeks=w)
            end = start + timedelta(days=6)
            weeks.append(f"W{w+1}")
            weekly_run.append(sum(
                s.trimp or 0 for s in self.optimizer.running_sessions
                if start <= s.session_date <= end))
            weekly_lift.append(sum(
                (s.tonnage or 0) / 100 for s in self.optimizer.lifting_sessions
                if start <= s.session_date <= end))
        x = np.arange(len(weeks))
        ax.bar(x - 0.15, weekly_run, 0.3, label="Running (TRIMP)",
               color=self.ACCENT, alpha=0.85)
        ax.bar(x + 0.15, weekly_lift, 0.3, label="Lifting (tonnage/100)",
               color=self.ORANGE, alpha=0.85)
        ax.set_xticks(x[::2]); ax.set_xticklabels(weeks[::2], fontsize=7)
        ax.set_title("Weekly Training Volume", fontsize=11)
        ax.legend(fontsize=8, facecolor=self.BG2, edgecolor="#444",
                  labelcolor=self.FG)
        fig.tight_layout()
        self._embed_figure(parent, fig, row=2, col=0, colspan=4)

    def _build_goals(self, parent):
        parent.columnconfigure(0, weight=1); parent.rowconfigure(1, weight=1)
        ttk.Label(parent, text="Goals & Milestones (Section 1 + 14)",
                  style="Header.TLabel").grid(row=0, column=0, padx=10,
                                               pady=(10, 4), sticky="w")
        frame = ttk.Frame(parent)
        frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        frame.columnconfigure(0, weight=1)

        for i, goal in enumerate(self.optimizer.goals):
            gf = ttk.LabelFrame(frame, text=goal.description, padding=10)
            gf.grid(row=i * 2, column=0, sticky="ew", pady=6)
            gf.columnconfigure(1, weight=1)
            info = (f"Type: {goal.goal_type.name}   |   "
                    f"Current: {goal.current_value}   |   "
                    f"Target: {goal.target_value}   |   "
                    f"Deadline: {goal.target_date}")
            ttk.Label(gf, text=info).grid(row=0, column=0, columnspan=3,
                                           sticky="w")
            if goal.milestones:
                cols = ("Date", "Target Value", "Label")
                tree = ttk.Treeview(gf, columns=cols, show="headings",
                                     height=4)
                for c in cols:
                    tree.heading(c, text=c); tree.column(c, width=180)
                for ms in goal.milestones:
                    tree.insert("", "end", values=(
                        ms["date"], ms["target_value"], ms["label"]))
                tree.grid(row=1, column=0, columnspan=3, sticky="ew",
                          pady=(6, 0))

        if len(self.optimizer.goals) >= 2:
            conflict = GoalConflictDetector.check_conflict(
                self.optimizer.goals[0], self.optimizer.goals[1])
            cf = ttk.LabelFrame(frame,
                                text="Goal Conflict Analysis (Section 14)",
                                padding=10)
            cf.grid(row=10, column=0, sticky="ew", pady=10)
            ttk.Label(cf,
                      text=f"Severity: {conflict['conflict_label'].upper()}",
                      font=("Segoe UI", 11, "bold"),
                      foreground=(self.YELLOW
                                  if conflict["conflict_severity"] >= 2
                                  else self.GREEN)).pack(anchor="w")
            ttk.Label(cf, text=conflict["recommendation"]).pack(
                anchor="w", pady=2)

    def _build_running_log(self, parent):
        parent.columnconfigure(0, weight=1); parent.rowconfigure(1, weight=1)
        ttk.Label(parent, text="Running Training Log (Section 2)",
                  style="Header.TLabel").grid(row=0, column=0, padx=10,
                                               pady=(10, 4), sticky="w")
        cols = ("Date", "Type", "Duration", "Distance", "HR", "RPE", "TRIMP")
        tree = ttk.Treeview(parent, columns=cols, show="headings", height=25)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120 if c != "Type" else 100)
        tree.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        sb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.grid(row=1, column=1, sticky="ns")
        for s in sorted(self.optimizer.running_sessions,
                        key=lambda x: x.session_date, reverse=True):
            tree.insert("", "end", values=(
                s.session_date, s.workout_type.name,
                f"{s.duration_minutes:.0f} min",
                f"{s.distance_km:.1f} km" if s.distance_km else "—",
                s.avg_heart_rate or "—", s.rpe or "—",
                f"{s.trimp:.1f}" if s.trimp else "—"))

    def _build_lifting_log(self, parent):
        parent.columnconfigure(0, weight=1); parent.rowconfigure(1, weight=1)
        ttk.Label(parent, text="Lifting Training Log (Section 3)",
                  style="Header.TLabel").grid(row=0, column=0, padx=10,
                                               pady=(10, 4), sticky="w")
        cols = ("Date", "Exercises", "Total Tonnage", "RPE")
        tree = ttk.Treeview(parent, columns=cols, show="headings", height=25)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=200 if c == "Exercises" else 140)
        tree.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        sb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.grid(row=1, column=1, sticky="ns")
        for s in sorted(self.optimizer.lifting_sessions,
                        key=lambda x: x.session_date, reverse=True):
            ex_names = ", ".join(e.exercise_name for e in s.exercises)
            tree.insert("", "end", values=(
                s.session_date, ex_names,
                f"{s.tonnage:,.0f} lbs" if s.tonnage else "—",
                s.session_rpe or "—"))

    def _build_assessments(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure((1, 2), weight=1)
        ttk.Label(parent, text="Assessments & Normalization (Section 5)",
                  style="Header.TLabel").grid(row=0, column=0, padx=10,
                                               pady=(10, 4), sticky="w")
        cols = ("Date", "Type", "Raw Value", "Unit",
                "Normalized (VDOT/1RM)", "Z-Score")
        tree = ttk.Treeview(parent, columns=cols, show="headings", height=12)
        for c in cols:
            tree.heading(c, text=c); tree.column(c, width=140)
        tree.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        for a in sorted(self.optimizer.assessments,
                        key=lambda x: x.assessment_date):
            tree.insert("", "end", values=(
                a.assessment_date, a.assessment_type, a.raw_value,
                a.raw_unit, a.normalized_score or "—",
                f"{a.z_score:.3f}" if a.z_score else "—"))
        fig, ax = self._make_fig(10, 3)
        dates = [a.assessment_date for a in self.optimizer.assessments]
        scores = [a.normalized_score for a in self.optimizer.assessments]
        ax.plot(dates, scores, "o-", color=self.ACCENT, markersize=5,
                linewidth=1.5)
        ax.set_title("Assessment Scores Over Time (VDOT)", fontsize=10)
        ax.set_ylabel("VDOT", fontsize=9)
        fig.autofmt_xdate(); fig.tight_layout()
        self._embed_figure(parent, fig, row=2, col=0)

    def _build_predictions(self, parent):
        parent.columnconfigure(0, weight=1); parent.rowconfigure(2, weight=1)
        ttk.Label(parent,
                  text="Prediction & Projection Engine (Section 6)",
                  style="Header.TLabel").grid(row=0, column=0, padx=10,
                                               pady=(10, 4), sticky="w")
        try:
            fit_result = self.optimizer.prediction_engine.fit(
                self.optimizer.assessments)
            pred = self.optimizer.prediction_engine.predict(
                target_date=self.optimizer.goals[0].target_date,
                base_date=self.optimizer.assessments[0].assessment_date,
                target_value=55.0)

            info_frame = ttk.Frame(parent)
            info_frame.grid(row=1, column=0, sticky="ew", padx=10)
            info_frame.columnconfigure((0, 1, 2, 3), weight=1)
            self._card(info_frame, 0, 0, "IMPROVEMENT SLOPE",
                       f"{fit_result['slope']:.4f}/day")
            self._card(info_frame, 0, 1, "R²",
                       f"{fit_result['r_squared']:.3f}")
            self._card(info_frame, 0, 2, "GOAL PROBABILITY",
                       f"{pred['goal_probability'] * 100:.1f}%",
                       "P(VDOT ≥ 55)")
            self._card(info_frame, 0, 3, "PERSONALIZATION",
                       f"{fit_result['personalization_pct']}%")

            fig, ax = self._make_fig(10, 4)
            base_date = self.optimizer.assessments[0].assessment_date
            ax_x = [(a.assessment_date - base_date).days
                    for a in self.optimizer.assessments]
            ax_y = [a.normalized_score for a in self.optimizer.assessments]
            ax.scatter(ax_x, ax_y, color=self.ACCENT, zorder=5, s=30,
                       label="Assessments")
            future_days = (self.optimizer.goals[0].target_date
                           - base_date).days
            x_line = np.linspace(0, future_days, 200)
            y_line = fit_result["slope"] * x_line + fit_result["intercept"]
            pred_std = np.sqrt(
                fit_result["slope_std"]**2 * x_line**2
                + self.optimizer.prediction_engine.residual_std**2)
            ax.plot(x_line, y_line, color=self.GREEN, linewidth=1.5,
                    label="Trend")
            ax.fill_between(x_line, y_line - 1.96 * pred_std,
                            y_line + 1.96 * pred_std,
                            alpha=0.15, color=self.GREEN, label="95% CI")
            ax.fill_between(x_line, y_line - pred_std,
                            y_line + pred_std,
                            alpha=0.2, color=self.GREEN, label="68% CI")
            ax.axhline(55, color=self.RED, linestyle="--", alpha=0.7,
                       label="Goal (VDOT 55)")
            ax.set_xlabel("Days from first assessment")
            ax.set_ylabel("VDOT")
            ax.set_title("Improvement Trajectory with Confidence Intervals")
            ax.legend(fontsize=7, facecolor=self.BG2, edgecolor="#444",
                      labelcolor=self.FG)
            fig.tight_layout()
            self._embed_figure(parent, fig, row=2, col=0)
        except Exception as e:
            ttk.Label(parent, text=f"Prediction error: {e}").grid(
                row=1, column=0)

    def _build_correlations(self, parent):
        parent.columnconfigure(0, weight=1); parent.rowconfigure(2, weight=1)
        ttk.Label(parent,
                  text="Correlation & Insight Engine (Section 7)",
                  style="Header.TLabel").grid(row=0, column=0, padx=10,
                                               pady=(10, 4), sticky="w")
        results_text = []
        metrics_to_test = ["sleep", "stress", "soreness", "mood", "energy"]
        p_values, all_results = [], []
        perf_daily = np.interp(np.arange(168), self.assess_days,
                               self.assess_scores)

        for mname in metrics_to_test:
            arr = self.metric_arrays.get(mname, np.array([]))
            if len(arr) < 30:
                continue
            n = min(len(arr), len(perf_daily))
            cr = CorrelationEngine.lagged_cross_correlation(
                arr[:n], perf_daily[:n], max_lag=14)
            if "error" in cr:
                continue
            gr = CorrelationEngine.granger_causality_test(
                arr[:n], perf_daily[:n])
            gs = gr.get("is_significant", False)
            insight = CorrelationEngine.generate_insight(
                mname, cr["best_lag"], cr["best_r"], cr["best_p"], gs)
            p_values.append(cr["best_p"])
            all_results.append({
                "metric": mname, "lag": cr["best_lag"], "r": cr["best_r"],
                "p": cr["best_p"], "granger_sig": gs,
                "granger_var": gr.get("variance_explained_pct", 0),
                "insight": insight, "correlations": cr["correlations"]})

        survives = (CorrelationEngine.benjamini_hochberg(p_values)
                    if p_values else [])

        results_text.append("=" * 60)
        results_text.append(
            "LAGGED CROSS-CORRELATION + GRANGER CAUSALITY RESULTS")
        results_text.append(
            f"Benjamini-Hochberg FDR correction applied "
            f"({len(p_values)} tests)")
        results_text.append("=" * 60)
        for i, res in enumerate(all_results):
            bh = ("✓ SURVIVES FDR"
                  if i < len(survives) and survives[i]
                  else "✗ Rejected by FDR")
            results_text.append(f"\n--- {res['metric'].upper()} ---")
            results_text.append(
                f"  Best lag: {res['lag']} days  |  r = {res['r']:.3f}  "
                f"|  p = {res['p']:.5f}  [{bh}]")
            results_text.append(
                f"  Granger: "
                f"{'Significant' if res['granger_sig'] else 'Not significant'}"
                f" (+{res['granger_var']:.1f}% variance)")
            if res["insight"]:
                results_text.append(f"  >> {res['insight']}")

        self._text_panel(parent, "\n".join(results_text), row=1, col=0, h=10)

        n_plots = len(all_results)
        if n_plots > 0:
            fig2 = Figure(figsize=(12, 3.5), dpi=100, facecolor=self.BG)
            for idx, res in enumerate(all_results):
                ax = fig2.add_subplot(1, n_plots, idx + 1)
                ax.set_facecolor(self.BG2)
                lags = [c[0] for c in res["correlations"]]
                rs = [c[1] for c in res["correlations"]]
                colors = [self.ACCENT
                          if abs(r) == max(abs(rr) for rr in rs)
                          else "#666" for r in rs]
                ax.bar(lags, rs, color=colors, alpha=0.8)
                ax.set_title(res["metric"], fontsize=9, color=self.ACCENT)
                ax.set_xlabel("Lag (days)", fontsize=7, color=self.FG)
                ax.set_ylabel("r", fontsize=7, color=self.FG)
                ax.tick_params(colors=self.FG, labelsize=7)
                ax.axhline(0, color="#555", linewidth=0.5)
                for spine in ax.spines.values():
                    spine.set_color("#444")
            fig2.tight_layout()
            self._embed_figure(parent, fig2, row=2, col=0)

    def _build_simulator(self, parent):
        parent.columnconfigure(0, weight=1); parent.rowconfigure(3, weight=1)
        ttk.Label(parent,
                  text="Training Simulator / What-If Engine (Section 9)",
                  style="Header.TLabel").grid(row=0, column=0, padx=10,
                                               pady=(10, 4), sticky="w")
        ctrl = ttk.LabelFrame(parent, text="Propose a Session", padding=8)
        ctrl.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        ttk.Label(ctrl, text="Days from now:").grid(row=0, column=0, padx=4)
        day_var = tk.IntVar(value=3)
        ttk.Spinbox(ctrl, from_=1, to=60, textvariable=day_var,
                     width=6).grid(row=0, column=1, padx=4)
        ttk.Label(ctrl, text="Load (TRIMP):").grid(row=0, column=2, padx=4)
        load_var = tk.DoubleVar(value=80.0)
        ttk.Spinbox(ctrl, from_=10, to=300, textvariable=load_var,
                     width=8, increment=10).grid(row=0, column=3, padx=4)

        result_frame = ttk.Frame(parent)
        result_frame.grid(row=2, column=0, sticky="ew", padx=10)
        result_frame.columnconfigure((0, 1, 2, 3), weight=1)

        chart_frame = ttk.Frame(parent)
        chart_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=4)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)

        def run_simulation():
            for w in result_frame.winfo_children():
                w.destroy()
            for w in chart_frame.winfo_children():
                w.destroy()
            proposed_day = float(self.session_days[-1]) + day_var.get()
            proposed_load = load_var.get()
            sim = self.optimizer.banister.simulate_session(
                self.session_days, self.session_loads,
                proposed_day, proposed_load, horizon_days=40)
            q_color = ("Green.TLabel" if sim["adaptation_quality"] > 0.7
                       else "Yellow.TLabel"
                       if sim["adaptation_quality"] > 0.4
                       else "Red.TLabel")
            self._card(result_frame, 0, 0, "ADAPTATION QUALITY",
                       f"{sim['adaptation_quality']:.0%}",
                       sim["overreaching"], q_color)
            self._card(result_frame, 0, 1, "PEAK PERFORMANCE",
                       f"{sim['peak_performance']:.1f}",
                       f"Day +{sim['optimal_day_offset']}")
            self._card(result_frame, 0, 2, "REMOVING BETTER?",
                       "YES" if sim["removing_session_better"] else "NO",
                       ("Fewer sessions = more gains"
                        if sim["removing_session_better"]
                        else "Session is beneficial"))
            acwr = BanisterModel.compute_acwr(self.daily_loads)
            self._card(result_frame, 0, 3, "CURRENT ACWR",
                       acwr["acwr"], acwr["zone"].upper())
            fig, ax = self._make_fig(11, 4)
            t = sim["time_points"]
            ax.plot(t, sim["performance_without"], "--", color="#888",
                    linewidth=1, label="Without session", alpha=0.7)
            ax.plot(t, sim["performance_with"], "-", color=self.ACCENT,
                    linewidth=2, label="With session")
            ax.fill_between(t, sim["performance_without"],
                            sim["performance_with"],
                            alpha=0.1, color=self.ACCENT)
            ax.axvline(proposed_day, color=self.YELLOW, linestyle=":",
                       alpha=0.8, label="Proposed session")
            peak_t = t[np.argmax(sim["performance_with"])]
            ax.axvline(peak_t, color=self.GREEN, linestyle=":", alpha=0.6,
                       label="Predicted peak")
            ax.set_xlabel("Day"); ax.set_ylabel("Predicted Performance")
            ax.set_title("What-If Simulation: Performance Response Curve")
            ax.legend(fontsize=8, facecolor=self.BG2, edgecolor="#444",
                      labelcolor=self.FG)
            fig.tight_layout()
            self._embed_figure(chart_frame, fig, row=0, col=0)

        ttk.Button(ctrl, text="▶  Simulate",
                   command=run_simulation).grid(row=0, column=4, padx=12)
        run_simulation()

    def _build_readiness(self, parent):
        parent.columnconfigure(0, weight=1); parent.rowconfigure(2, weight=1)
        ttk.Label(parent, text="Daily Readiness Score (Section 11)",
                  style="Header.TLabel").grid(row=0, column=0, padx=10,
                                               pady=(10, 4), sticky="w")
        ctrl = ttk.LabelFrame(parent, text="Adjust Today's Inputs",
                              padding=8)
        ctrl.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        vars_ = {}
        defaults = {"sleep": 7.5, "fatigue": 30, "days_since_hard": 2,
                     "acwr": 1.1, "stress": 4.0, "soreness": 3.0}
        for i, (name, default) in enumerate(defaults.items()):
            ttk.Label(ctrl, text=f"{name}:").grid(row=0, column=i*2, padx=4)
            v = tk.DoubleVar(value=default); vars_[name] = v
            ttk.Spinbox(ctrl, from_=0, to=200, textvariable=v, width=6,
                         increment=0.5).grid(row=0, column=i*2+1, padx=2)

        result_area = ttk.Frame(parent)
        result_area.grid(row=2, column=0, sticky="nsew", padx=10, pady=4)
        result_area.columnconfigure((0, 1), weight=1)
        result_area.rowconfigure(0, weight=1)

        def compute_readiness():
            for w in result_area.winfo_children():
                w.destroy()
            inputs = {k: v.get() for k, v in vars_.items()}
            result = self.optimizer.daily_readiness(inputs)
            band_colors = {
                ReadinessBand.GREEN: self.GREEN,
                ReadinessBand.YELLOW: self.YELLOW,
                ReadinessBand.ORANGE: self.ORANGE,
                ReadinessBand.RED: self.RED}
            color = band_colors.get(result["band"], self.FG)
            sf = ttk.Frame(result_area, style="Card.TFrame", padding=20)
            sf.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
            ttk.Label(sf, text="READINESS SCORE", style="Card.TLabel",
                      font=("Segoe UI", 10)).pack(pady=(0, 4))
            tk.Label(sf, text=str(result["score"]),
                     font=("Segoe UI", 72, "bold"),
                     fg=color, bg=self.CARD_BG).pack()
            tk.Label(sf, text=result["band"].value,
                     font=("Segoe UI", 16),
                     fg=color, bg=self.CARD_BG).pack()
            cf = ttk.Frame(result_area, style="Card.TFrame", padding=12)
            cf.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
            ttk.Label(cf, text="COMPONENT BREAKDOWN", style="Card.TLabel",
                      font=("Segoe UI", 10, "bold")).pack(anchor="w",
                                                           pady=(0, 8))
            for cname, data in result["components"].items():
                rf = ttk.Frame(cf, style="Card.TFrame")
                rf.pack(fill="x", pady=2)
                ttk.Label(rf, text=f"{cname}:", style="Card.TLabel",
                          width=16).pack(side="left")
                bar_frame = tk.Frame(rf, bg="#333", height=16, width=200)
                bar_frame.pack(side="left", padx=4)
                bar_frame.pack_propagate(False)
                bar_w = int(data["normalized"] * 200)
                bar_color = (self.GREEN if data["normalized"] > 0.7
                             else self.YELLOW if data["normalized"] > 0.4
                             else self.RED)
                tk.Frame(bar_frame, bg=bar_color,
                         width=bar_w, height=16).place(x=0, y=0)
                ttk.Label(rf,
                          text=f"{data['raw']} ({data['normalized']:.0%})",
                          style="Card.TLabel").pack(side="left", padx=6)
                ttk.Label(rf, text=f"w={data['weight']:.2f}",
                          style="Card.TLabel",
                          foreground="#888").pack(side="right")

        ttk.Button(ctrl, text="▶  Compute",
                   command=compute_readiness).grid(
            row=0, column=len(defaults)*2, padx=12)
        compute_readiness()

    def _build_injury_risk(self, parent):
        parent.columnconfigure(0, weight=1); parent.rowconfigure(2, weight=1)
        ttk.Label(parent,
                  text="Injury Risk Prediction (Section 12) — "
                       "Population-Based",
                  style="Header.TLabel").grid(row=0, column=0, padx=10,
                                               pady=(10, 4), sticky="w")
        ms = InjuryRiskModel.compute_monotony_strain(self.daily_loads[-7:])
        risk = InjuryRiskModel.predict_injury_risk(
            self.acwr_result["acwr"], ms["monotony"], ms["strain"],
            fatigue=self.current_fatigue,
            soreness=self.current_soreness,
            readiness_trend=self.optimizer.recent_readiness_trend(days=3),
        )

        info = ttk.Frame(parent)
        info.grid(row=1, column=0, sticky="ew", padx=10)
        info.columnconfigure((0, 1, 2, 3), weight=1)
        r_color = ("Green.TLabel" if risk["risk_pct"] < 20
                   else "Yellow.TLabel" if risk["risk_pct"] < 35
                   else "Red.TLabel")
        self._card(info, 0, 0, "INJURY RISK", f"{risk['risk_pct']}%",
                   risk["risk_level"].upper(), r_color)
        self._card(info, 0, 1, "MONOTONY", f"{ms['monotony']:.2f}",
                   "> 2.0 = HIGH" if ms["monotony"] > 2 else "Normal")
        self._card(info, 0, 2, "STRAIN", f"{ms['strain']:.0f}")
        self._card(info, 0, 3, "HARD SESSIONS BLOCKED",
                   "YES" if risk["block_hard_sessions"] else "NO",
                   "Safety cap active" if risk["block_hard_sessions"] else "")

        fig = Figure(figsize=(12, 3.5), dpi=100, facecolor=self.BG)
        ax1 = fig.add_subplot(121); ax1.set_facecolor(self.BG2)
        factors = [f[0] for f in risk["contributing_factors"]]
        values = [f[1] for f in risk["contributing_factors"]]
        colors = [self.RED if v > 0 else self.GREEN for v in values]
        ax1.barh(factors, values, color=colors, alpha=0.8)
        ax1.set_title("Risk Factor Contributions", fontsize=9,
                      color=self.ACCENT)
        ax1.tick_params(colors=self.FG, labelsize=8)
        for sp in ax1.spines.values():
            sp.set_color("#444")

        ax2 = fig.add_subplot(122); ax2.set_facecolor(self.BG2)
        acwr_history = []
        for i in range(28, len(self.daily_loads)):
            a = BanisterModel.compute_acwr(self.daily_loads[:i + 1])
            acwr_history.append(a["acwr"])
        ax2.plot(acwr_history, color=self.ACCENT, linewidth=1.5)
        ax2.axhline(1.3, color=self.YELLOW, ls="--", alpha=0.7,
                    label="Caution (1.3)")
        ax2.axhline(1.5, color=self.RED, ls="--", alpha=0.7,
                    label="High risk (1.5)")
        ax2.axhline(0.8, color=self.GREEN, ls="--", alpha=0.5,
                    label="Safe floor (0.8)")
        ax2.fill_between(range(len(acwr_history)), 0.8, 1.3, alpha=0.08,
                         color=self.GREEN)
        ax2.set_title("ACWR Over Time (EWMA)", fontsize=9,
                      color=self.ACCENT)
        ax2.legend(fontsize=7, facecolor=self.BG2, edgecolor="#444",
                   labelcolor=self.FG)
        ax2.tick_params(colors=self.FG, labelsize=8)
        for sp in ax2.spines.values():
            sp.set_color("#444")
        fig.tight_layout()
        self._embed_figure(parent, fig, row=2, col=0)

    def _build_adaptive_plan(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure((1, 3), weight=1)
        ttk.Label(parent, text="Adaptive Training Plan (Section 8)",
                  style="Header.TLabel").grid(row=0, column=0, padx=10,
                                               pady=(10, 4), sticky="w")
        suggestions = AdaptivePlan.rule_based_suggestions(
            recent_sessions=[], readiness_score=75,
            injury_risk=0.15, acwr=self.acwr_result["acwr"],
            adaptation_quality=0.72, days_since_hard=2)
        sf = ttk.LabelFrame(parent,
                            text="Phase 1: Rule-Based Suggestions (Today)",
                            padding=10)
        sf.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        for s in suggestions:
            color = (self.GREEN if s["type"] in ("hard",)
                     else self.YELLOW if s["type"] in ("moderate",)
                     else self.ORANGE if s["type"] in ("easy",)
                     else self.RED)
            rf = ttk.Frame(sf); rf.pack(fill="x", pady=4)
            tk.Label(rf, text=f"  {s['type'].upper()}  ",
                     font=("Segoe UI", 11, "bold"), fg="white", bg=color,
                     padx=8, pady=2).pack(side="left")
            ttk.Label(rf, text=s["reason"],
                      wraplength=800).pack(side="left", padx=10)

        ttk.Label(parent, text="Phase 2: Thompson Sampling Bandit",
                  style="SubHeader.TLabel").grid(row=2, column=0, padx=10,
                                                  pady=(10, 2), sticky="w")
        bandit = self.optimizer.bandit
        rng = np.random.RandomState(123)
        reward_probs = {"easy_run": 0.3, "intervals": 0.6, "tempo": 0.55,
                        "long_run": 0.5, "hill_repeats": 0.45,
                        "heavy_lift": 0.35, "moderate_lift": 0.4,
                        "rest": 0.5}
        for _ in range(100):
            arm, _ = bandit.sample_and_recommend()
            bandit.update(arm, rng.random() < reward_probs.get(arm, 0.4))
        bs = bandit.get_stats()
        fig, ax = self._make_fig(10, 4)
        arms = sorted(bs.keys(), key=lambda a: bs[a]["mean"], reverse=True)
        means = [bs[a]["mean"] for a in arms]
        totals = [bs[a]["total_trials"] for a in arms]
        bars = ax.barh(arms, means, color=self.ACCENT, alpha=0.8)
        for bar, t in zip(bars, totals):
            ax.text(bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"n={t}", va="center", fontsize=8, color=self.FG)
        ax.set_xlabel("Estimated Success Rate", fontsize=9)
        ax.set_title("Thompson Sampling: Session Type Success Rates "
                     "(after 100 trials)", fontsize=10)
        ax.tick_params(colors=self.FG, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#444")
        fig.tight_layout()
        self._embed_figure(parent, fig, row=3, col=0)

    def _build_block_analysis(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure((1, 3), weight=1)
        ttk.Label(parent,
                  text="Retrospective Block Analysis "
                       "(Sections 16, 17, 18)",
                  style="Header.TLabel").grid(row=0, column=0, padx=10,
                                               pady=(10, 4), sticky="w")

        block_a = self.assess_scores[:6]
        block_b = self.assess_scores[6:]
        comparison = BlockAnalysis.compare_blocks(block_a, block_b)

        info = ttk.Frame(parent)
        info.grid(row=1, column=0, sticky="ew", padx=10)
        info.columnconfigure((0, 1, 2, 3), weight=1)
        d_color = "Green.TLabel" if comparison["cohens_d"] > 0 else "Red.TLabel"
        self._card(info, 0, 0, "COHEN'S D",
                   f"{comparison['cohens_d']:.3f}",
                   comparison["d_interpretation"].upper(), d_color)
        self._card(info, 0, 1, "P-VALUE",
                   f"{comparison['p_value']:.4f}",
                   "Significant" if comparison["significant"]
                   else "Not significant")
        self._card(info, 0, 2, "BLOCK A MEAN",
                   f"{np.mean(block_a):.2f}",
                   f"n={len(block_a)} assessments")
        self._card(info, 0, 3, "BLOCK B MEAN",
                   f"{np.mean(block_b):.2f}",
                   f"n={len(block_b)} assessments")

        changepoints = BlockAnalysis.detect_changepoints(
            self.daily_loads, threshold=3.0)
        plateau = PlateauDetector.detect_plateau(
            self.assess_scores, self.assess_days)

        fb = FeedbackLoop(); bias_history = []
        for _ in range(20):
            predicted = 6.5 + np.random.uniform(-0.5, 0.5)
            actual = 7.0 + np.random.uniform(-0.5, 0.5)
            result = fb.record_feedback(predicted, actual)
            bias_history.append(result["current_bias"])

        pm = PostMortem.analyze(
            assessment_score=50.5, predicted_score=49.0,
            metric_values_lead_up={
                "sleep": np.array([8.1, 8.0, 8.3]),
                "stress": np.array([3.0, 2.5, 3.2]),
                "soreness": np.array([2.0, 1.5, 2.0])},
            metric_norms={
                "sleep": (7.2, 0.8), "stress": (5.0, 1.5),
                "soreness": (4.0, 1.2)},
            correlation_coefficients={
                "sleep": 0.55, "stress": -0.40, "soreness": -0.35},
            optimal_lags={"sleep": 2, "stress": 1, "soreness": 1})

        txt = []
        txt.append("POST-SESSION FEEDBACK LOOP (Section 17):")
        txt.append(f"  After 20 sessions: bias = {fb.bias:.3f}")
        txt.append(f"  Bias detected: {fb.consecutive_biased >= 10}")
        txt.append(f"\nPLATEAU DETECTION (Section 10):")
        txt.append(f"  {plateau['message']}")
        txt.append(f"\nCHANGEPOINTS DETECTED: {len(changepoints)}")
        if changepoints:
            txt.append(f"  At day indices: {changepoints[:10]}")
        txt.append(f"\n{'='*50}")
        txt.append("RACE POST-MORTEM (Section 18):")
        txt.append(f"  {pm['summary']}")
        for c in pm["top_positive_contributors"]:
            txt.append(f"  + {c['metric']}: contribution = "
                       f"{c['contribution']:+.3f} "
                       f"(z = {c['deviation_z']:+.2f})")
        for c in pm["top_negative_contributors"]:
            txt.append(f"  - {c['metric']}: contribution = "
                       f"{c['contribution']:+.3f} "
                       f"(z = {c['deviation_z']:+.2f})")
        self._text_panel(parent, "\n".join(txt), row=2, col=0, h=12)

        fig = Figure(figsize=(12, 3.5), dpi=100, facecolor=self.BG)
        ax1 = fig.add_subplot(131); ax1.set_facecolor(self.BG2)
        ax1.boxplot(
            [block_a, block_b],
            labels=["Block A\n(Wk 1-12)", "Block B\n(Wk 13-24)"],
            patch_artist=True,
            boxprops=dict(facecolor=self.BG3, color=self.ACCENT),
            medianprops=dict(color=self.YELLOW),
            whiskerprops=dict(color=self.FG),
            capprops=dict(color=self.FG),
            flierprops=dict(markerfacecolor=self.RED))
        ax1.set_title("Block Comparison", fontsize=9, color=self.ACCENT)
        ax1.tick_params(colors=self.FG, labelsize=8)
        for sp in ax1.spines.values():
            sp.set_color("#444")

        ax2 = fig.add_subplot(132); ax2.set_facecolor(self.BG2)
        ax2.plot(self.daily_loads, color=self.ACCENT, alpha=0.6,
                 linewidth=0.8)
        for cp in changepoints:
            ax2.axvline(cp, color=self.RED, alpha=0.5, linewidth=1)
        ax2.set_title("Changepoints in Daily Load", fontsize=9,
                      color=self.ACCENT)
        ax2.tick_params(colors=self.FG, labelsize=8)
        for sp in ax2.spines.values():
            sp.set_color("#444")

        ax3 = fig.add_subplot(133); ax3.set_facecolor(self.BG2)
        ax3.plot(bias_history, color=self.ORANGE, linewidth=1.5)
        ax3.axhline(1.0, color=self.RED, ls="--", alpha=0.6,
                    label="Threshold")
        ax3.axhline(-1.0, color=self.RED, ls="--", alpha=0.6)
        ax3.axhline(0, color="#555", linewidth=0.5)
        ax3.set_title("Prediction Bias (EWMA)", fontsize=9,
                      color=self.ACCENT)
        ax3.legend(fontsize=7, facecolor=self.BG2, edgecolor="#444",
                   labelcolor=self.FG)
        ax3.tick_params(colors=self.FG, labelsize=8)
        for sp in ax3.spines.values():
            sp.set_color("#444")
        fig.tight_layout()
        self._embed_figure(parent, fig, row=3, col=0)

    def _build_data_quality(self, parent):
        parent.columnconfigure(0, weight=1); parent.rowconfigure(2, weight=1)
        ttk.Label(parent,
                  text="Data Quality Scoring (Section 15) + "
                       "Personalization (Section 10)",
                  style="Header.TLabel").grid(row=0, column=0, padx=10,
                                               pady=(10, 4), sticky="w")
        n_train_days = 168
        n_logged = len(set(
            s.session_date for s in self.optimizer.running_sessions
        ) | set(
            s.session_date for s in self.optimizer.lifting_sessions))
        logging_rate = n_logged / n_train_days
        n_metric_days = len(set(
            e.entry_date for e in self.optimizer.metric_entries))
        metric_rate = n_metric_days / n_train_days
        sleep_vals = self.metric_arrays.get("sleep", np.array([]))
        outliers = (DataQualityScorer.detect_outliers(sleep_vals)
                    if len(sleep_vals) > 5 else {"outlier_rate": 0})
        dq = DataQualityScorer.composite_score(
            logging_rate, metric_rate, 0.85, outliers["outlier_rate"])

        info = ttk.Frame(parent)
        info.grid(row=1, column=0, sticky="ew", padx=10)
        info.columnconfigure((0, 1, 2, 3, 4), weight=1)
        dq_color = ("Green.TLabel" if dq["score"] > 0.7
                    else "Yellow.TLabel")
        self._card(info, 0, 0, "DATA QUALITY", f"{dq['score']:.0%}",
                   f"Grade {dq['grade']}", dq_color)
        self._card(info, 0, 1, "LOGGING RATE", f"{logging_rate:.0%}")
        self._card(info, 0, 2, "METRIC RATE", f"{metric_rate:.0%}")
        self._card(info, 0, 3, "OUTLIER RATE",
                   f"{outliers['outlier_rate']:.1%}")
        status = self.optimizer.status()
        self._card(info, 0, 4, "PERSONALIZATION",
                   f"{status['personalization_pct']}%",
                   f"Stage {status['personalization_stage']}")

        tracker = self.optimizer.personalization
        txt = ["PERSONALIZATION WEIGHTS PER FEATURE:", ""]
        for feature in sorted(tracker.FEATURE_THRESHOLDS.keys()):
            w = tracker.get_weight(feature)
            bar = "█" * int(w * 20) + "░" * (20 - int(w * 20))
            txt.append(
                f"  {feature:<30} [{bar}] {w:.0%}  "
                f"({tracker.data_counts.get(feature, 0)}/"
                f"{tracker.FEATURE_THRESHOLDS[feature]})")
        txt.append(f"\n{'='*50}")
        txt.append("DATA QUALITY SUGGESTIONS:")
        for s in dq["suggestions"]:
            txt.append(f"  → {s}")
        if not dq["suggestions"]:
            txt.append("  ✓ Data quality is excellent. Keep it up!")
        txt.append(
            f"\nConfidence penalty applied: {dq['confidence_penalty']:.0%}")

        txt.append(f"\n{'='*50}")
        txt.append("PREDICTION CONFIDENCE DEMO (Section 9g):")
        scorer = SimilarityScorer(k=5)
        for i in range(30):
            scorer.add_session(
                {"type": "tempo",
                 "load": 70 + np.random.uniform(-10, 10),
                 "fatigue": 0.3 + np.random.uniform(-0.1, 0.1)},
                outcome=48 + np.random.uniform(-1, 1))
        conf = scorer.predict_confidence(
            {"type": "tempo", "load": 72, "fatigue": 0.32})
        txt.append(
            f"  Confidence: {conf['confidence'].value} "
            f"({conf['confidence_pct']:.0f}%)")
        txt.append(f"  {conf['reasoning']}")

        txt.append(f"\n{'='*50}")
        txt.append("MODEL DRIFT DETECTION (Section 10):")
        dd = DriftDetector(); drift_found = False
        for i in range(15):
            drift_offset = 2.0 if i >= 8 else 0
            res = dd.add_residual(
                50 + drift_offset, 50 + np.random.uniform(-0.5, 0.5))
            if res["drift_detected"]:
                drift_found = True
                txt.append(
                    f"  ⚠ Drift detected at observation {i+1}! "
                    f"Refit recommended.")
        if not drift_found:
            txt.append("  No drift detected in demo sequence.")

        txt.append(f"\n{'='*50}")
        txt.append("CONCURRENT TRAINING INTERFERENCE (Section 13):")
        with_lift = [45 + np.random.uniform(-2, 2) for _ in range(8)]
        without_lift = [48 + np.random.uniform(-2, 2) for _ in range(8)]
        interf = InterferenceDetector.detect_interference(
            with_lift, without_lift)
        txt.append(
            f"  Detected: {interf.get('interference_detected', 'N/A')}")
        txt.append(
            f"  Test: {interf.get('test_used', 'N/A')}, "
            f"d = {interf.get('effect_size_cohens_d', 'N/A')}, "
            f"p = {interf.get('p_value', 'N/A')}")
        self._text_panel(parent, "\n".join(txt), row=2, col=0, h=22)

    def _build_log_data(self, parent):
        """
        Forms for adding running/lifting sessions, assessments, and metric
        entries through the GUI. On successful submit each handler ingests
        the data into self.optimizer and calls self.refresh() so the rest
        of the dashboard reflects the change immediately.
        """
        parent.columnconfigure((0, 1), weight=1)
        parent.rowconfigure((1, 2), weight=1)

        ttk.Label(
            parent,
            text="Log Data — entries here update every other tab on submit",
            style="Header.TLabel",
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 4), sticky="w")

        self._log_status_var = tk.StringVar(value="Ready.")
        ttk.Label(parent, textvariable=self._log_status_var,
                  style="SubHeader.TLabel").grid(
            row=3, column=0, columnspan=2, padx=10, pady=(2, 8), sticky="w")

        self._build_running_form(parent, row=1, col=0)
        self._build_lifting_form(parent, row=1, col=1)
        self._build_assessment_form(parent, row=2, col=0)
        self._build_metric_form(parent, row=2, col=1)

    def _form_field(self, parent, row, label, var, width=22, hint=None):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=4, pady=2)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        if hint:
            ttk.Label(parent, text=hint, foreground="#888").grid(
                row=row, column=2, sticky="w", padx=4)
        return entry

    def _form_combo(self, parent, row, label, var, values, width=20):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=4, pady=2)
        combo = ttk.Combobox(parent, textvariable=var, values=values,
                             width=width, state="readonly")
        combo.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        return combo

    def _today_iso(self) -> str:
        latest = max(
            (s.session_date for s in self.optimizer.running_sessions),
            default=date(2025, 1, 1),
        )
        return (latest + timedelta(days=1)).isoformat()

    def _set_status(self, msg: str, ok: bool = True):
        self._log_status_var.set(("✓ " if ok else "✗ ") + msg)

    @staticmethod
    def _parse_iso_date(s: str) -> date:
        return date.fromisoformat(s.strip())

    @staticmethod
    def _parse_optional_float(s: str) -> Optional[float]:
        s = s.strip()
        return float(s) if s else None

    @staticmethod
    def _parse_optional_int(s: str) -> Optional[int]:
        s = s.strip()
        return int(s) if s else None

    def _build_running_form(self, parent, row, col):
        frame = ttk.LabelFrame(parent, text="Add Running Session", padding=10)
        frame.grid(row=row, column=col, sticky="nsew", padx=8, pady=4)
        frame.columnconfigure(1, weight=1)

        v_date = tk.StringVar(value=self._today_iso())
        v_type = tk.StringVar(value=RunWorkoutType.EASY.name)
        v_dur  = tk.StringVar()
        v_dist = tk.StringVar()
        v_ahr  = tk.StringVar()
        v_mhr  = tk.StringVar()
        v_rpe  = tk.StringVar()

        self._form_field(frame, 0, "Date (YYYY-MM-DD)", v_date)
        self._form_combo(frame, 1, "Workout type", v_type,
                         [t.name for t in RunWorkoutType])
        self._form_field(frame, 2, "Duration (min)", v_dur)
        self._form_field(frame, 3, "Distance (km)", v_dist, hint="optional")
        self._form_field(frame, 4, "Avg HR", v_ahr, hint="optional")
        self._form_field(frame, 5, "Max HR", v_mhr, hint="optional")
        self._form_field(frame, 6, "RPE (1-10)", v_rpe, hint="optional")

        def submit():
            try:
                session = RunningSession(
                    session_id=f"run_{len(self.optimizer.running_sessions) + 1}",
                    session_date=self._parse_iso_date(v_date.get()),
                    workout_type=RunWorkoutType[v_type.get()],
                    duration_minutes=float(v_dur.get()),
                    distance_km=self._parse_optional_float(v_dist.get()),
                    avg_heart_rate=self._parse_optional_int(v_ahr.get()),
                    max_heart_rate=self._parse_optional_int(v_mhr.get()),
                    rpe=self._parse_optional_float(v_rpe.get()),
                )
            except (ValueError, KeyError) as exc:
                messagebox.showerror("Invalid input", str(exc))
                return
            trimp = self.optimizer.add_running_session(session)
            self._set_status(
                f"Logged running session on {session.session_date} "
                f"(TRIMP={trimp:.1f}).")
            self.refresh()

        ttk.Button(frame, text="Add Running Session", command=submit).grid(
            row=7, column=0, columnspan=3, pady=(8, 0), sticky="ew")

    def _build_lifting_form(self, parent, row, col):
        frame = ttk.LabelFrame(parent, text="Add Lifting Session", padding=10)
        frame.grid(row=row, column=col, sticky="nsew", padx=8, pady=4)
        frame.columnconfigure(1, weight=1)

        v_date = tk.StringVar(value=self._today_iso())
        v_name = tk.StringVar(value="squat")
        v_sets = tk.StringVar(value="5")
        v_reps = tk.StringVar(value="5,5,5,5,5")
        v_wt   = tk.StringVar()
        v_srpe = tk.StringVar()

        self._form_field(frame, 0, "Date (YYYY-MM-DD)", v_date)
        self._form_field(frame, 1, "Exercise name", v_name)
        self._form_field(frame, 2, "Sets", v_sets)
        self._form_field(frame, 3, "Reps per set", v_reps,
                         hint="comma-separated")
        self._form_field(frame, 4, "Weight", v_wt)
        self._form_field(frame, 5, "Session RPE", v_srpe, hint="optional")

        def submit():
            try:
                reps_list = [int(x) for x in v_reps.get().split(",") if x.strip()]
                exercise = LiftingExercise(
                    exercise_name=v_name.get().strip() or "exercise",
                    muscle_groups=[],
                    sets=int(v_sets.get()),
                    reps_per_set=reps_list,
                    weight=float(v_wt.get()),
                )
                session = LiftingSession(
                    session_id=f"lift_{len(self.optimizer.lifting_sessions) + 1}",
                    session_date=self._parse_iso_date(v_date.get()),
                    exercises=[exercise],
                    session_rpe=self._parse_optional_float(v_srpe.get()),
                )
            except (ValueError, KeyError) as exc:
                messagebox.showerror("Invalid input", str(exc))
                return
            tonnage = self.optimizer.add_lifting_session(session)
            self._set_status(
                f"Logged lifting session on {session.session_date} "
                f"(tonnage={tonnage:.0f}).")
            self.refresh()

        ttk.Button(frame, text="Add Lifting Session", command=submit).grid(
            row=6, column=0, columnspan=3, pady=(8, 0), sticky="ew")

    def _build_assessment_form(self, parent, row, col):
        frame = ttk.LabelFrame(parent, text="Add Assessment", padding=10)
        frame.grid(row=row, column=col, sticky="nsew", padx=8, pady=4)
        frame.columnconfigure(1, weight=1)

        v_date = tk.StringVar(value=self._today_iso())
        v_dom  = tk.StringVar(value="running")
        v_type = tk.StringVar(value="5K")
        v_val  = tk.StringVar()
        v_unit = tk.StringVar(value="seconds")

        self._form_field(frame, 0, "Date (YYYY-MM-DD)", v_date)
        self._form_combo(frame, 1, "Domain", v_dom, ["running", "lifting"])
        self._form_field(frame, 2, "Assessment type", v_type,
                         hint='e.g. "5K" or "1RM_squat"')
        self._form_field(frame, 3, "Raw value", v_val)
        self._form_field(frame, 4, "Raw unit", v_unit,
                         hint='"seconds" / "lbs" / "kg"')

        def submit():
            try:
                assessment = Assessment(
                    assessment_id=f"assess_{len(self.optimizer.assessments) + 1}",
                    assessment_date=self._parse_iso_date(v_date.get()),
                    domain=v_dom.get().strip(),
                    raw_value=float(v_val.get()),
                    raw_unit=v_unit.get().strip(),
                    assessment_type=v_type.get().strip(),
                )
            except ValueError as exc:
                messagebox.showerror("Invalid input", str(exc))
                return
            self.optimizer.add_assessment(assessment)
            score = assessment.normalized_score
            self._set_status(
                f"Logged assessment on {assessment.assessment_date} "
                f"(normalized={score:.2f})." if score is not None
                else f"Logged assessment on {assessment.assessment_date}.")
            self.refresh()

        ttk.Button(frame, text="Add Assessment", command=submit).grid(
            row=5, column=0, columnspan=3, pady=(8, 0), sticky="ew")

    def _build_metric_form(self, parent, row, col):
        frame = ttk.LabelFrame(parent, text="Add Metric Entry", padding=10)
        frame.grid(row=row, column=col, sticky="nsew", padx=8, pady=4)
        frame.columnconfigure(1, weight=1)

        v_date = tk.StringVar(value=self._today_iso())
        v_id   = tk.StringVar(value="sleep")
        v_val  = tk.StringVar()

        self._form_field(frame, 0, "Date (YYYY-MM-DD)", v_date)
        self._form_combo(frame, 1, "Metric", v_id,
                         ["sleep", "stress", "soreness", "mood", "energy"])
        self._form_field(frame, 2, "Value", v_val)

        def submit():
            try:
                entry = MetricEntry(
                    metric_id=v_id.get().strip(),
                    entry_date=self._parse_iso_date(v_date.get()),
                    value=float(v_val.get()),
                )
            except ValueError as exc:
                messagebox.showerror("Invalid input", str(exc))
                return
            self.optimizer.metric_entries.append(entry)
            self._set_status(
                f"Logged {entry.metric_id} = {entry.value} on {entry.entry_date}.")
            self.refresh()

        ttk.Button(frame, text="Add Metric Entry", command=submit).grid(
            row=3, column=0, columnspan=3, pady=(8, 0), sticky="ew")


"""
====================================================================
ENTRY POINT
====================================================================
main() launches the Tkinter GUI. Tkinter is required to run this
application; if it isn't available in the active Python environment
the script exits with a clear error message rather than a stack trace.
"""

def main():
    if not _HAS_GUI:
        print("ERROR: tkinter is not available in this environment.")
        print("Install tkinter (e.g. `sudo apt install python3-tk`)")
        print("or run from a Python build that bundles it.")
        raise SystemExit(1)
    root = tk.Tk()
    FitnessOptimizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
