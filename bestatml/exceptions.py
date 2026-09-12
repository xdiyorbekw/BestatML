from __future__ import annotations

from sklearn.exceptions import NotFittedError as SklearnNotFittedError


class BestatMLError(Exception):
    """Base exception for BestatML-specific failures."""


class BestatMLDataError(BestatMLError, ValueError):
    """Raised when training or inference data is invalid."""


class BestatMLValidationError(BestatMLError, ValueError):
    """Raised when validly typed input violates a BestatML constraint."""


class BestatMLNotFittedError(BestatMLError, SklearnNotFittedError):
    """Raised when a fitted estimator is required but has not been fitted."""


class BestatMLConfigurationError(BestatMLError, ValueError):
    """Raised when estimator configuration is invalid or contradictory."""


class BestatMLTrainingError(BestatMLError, RuntimeError):
    """Raised when BestatML cannot complete training successfully."""


class BestatMLEnsembleError(BestatMLError, RuntimeError):
    """Raised when an ensemble cannot be constructed safely."""


class BestatMLPersistenceError(BestatMLError, OSError):
    """Raised when a BestatML artifact cannot be saved or loaded."""


class BestatMLDependencyError(BestatMLError, ImportError):
    """Raised when an explicitly requested optional dependency is unavailable."""


# Compatibility aliases retained for the previous API.
DataValidationError = BestatMLDataError
ConfigurationError = BestatMLConfigurationError
PersistenceError = BestatMLPersistenceError
CandidateModelError = BestatMLTrainingError

__all__ = [
    "BestatMLError",
    "BestatMLDataError",
    "BestatMLValidationError",
    "BestatMLNotFittedError",
    "BestatMLConfigurationError",
    "BestatMLTrainingError",
    "BestatMLEnsembleError",
    "BestatMLPersistenceError",
    "BestatMLDependencyError",
    "DataValidationError",
    "ConfigurationError",
    "PersistenceError",
    "CandidateModelError",
]
