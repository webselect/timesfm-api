"""Configuration du service, lue depuis l'environnement (ou un fichier .env)."""

from __future__ import annotations

import functools
import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Constantes imposees par TimesFM 3.0 (cf. src/timesfm3/timesfm3_forecaster.py).
MAX_CONTEXT_LENGTH = 15360
INPUT_PATCH_LENGTH = 32
OUTPUT_PATCH_LENGTH = 64
# Le modele ne traite pas plus de 32 variables (cibles + covariables) par passe ;
# au-dela, TimesFM3Evaluator decoupe automatiquement.
MAX_VARIATES_PER_FORWARD = 32
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MEDIAN_QUANTILE_INDEX = 4


class Settings(BaseSettings):
    """Parametres du service.

    Les champs `TIMESFM_*` correspondent un pour un a `timesfm3.ModelConfig` : ils sont
    consommes a la construction du modele et ne peuvent donc pas varier d'une requete a
    l'autre. Ils sont renvoyes tels quels par `GET /v1/model`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Modele -------------------------------------------------------------
    timesfm_checkpoint: str = "google/timesfm-3.0-pytorch"
    timesfm_device: str = "auto"
    timesfm_per_core_batch_size: int = 4
    timesfm_revision: str | None = None
    timesfm_cache_dir: str | None = None
    timesfm_local_files_only: bool = False
    hf_token: str | None = None

    # --- Options de construction du modele ----------------------------------
    timesfm_use_stitching: bool = True
    timesfm_use_linear_detrending: bool = True
    timesfm_linear_detrending_threshold: float = 0.5
    timesfm_use_iterative_cpm_revin: bool = True
    timesfm_use_frozen_running_stats: bool = False
    timesfm_use_variate_attention: bool = True
    timesfm_use_sdpa: bool = True
    timesfm_value_clip: float = 1e20
    timesfm_input_transform: str = "identity"

    # --- Garde-fous de l'API ------------------------------------------------
    api_max_series: int = Field(default=32, ge=1)
    api_max_horizon: int = Field(default=1024, ge=1)
    api_max_variates: int = Field(default=64, ge=1)
    api_preload: bool = True
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    def resolve_device(self) -> str:
        """Resout `auto` en un device PyTorch concret."""
        if self.timesfm_device != "auto":
            return self.timesfm_device
        try:
            import torch
        except ImportError:  # pragma: no cover - torch est une dependance dure
            return "cpu"
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def model_kwargs(self) -> dict:
        """Arguments passes a `timesfm3.ModelConfig`."""
        return {
            "checkpoint_path": self.timesfm_checkpoint,
            "per_core_batch_size": self.timesfm_per_core_batch_size,
            "use_stitching": self.timesfm_use_stitching,
            "use_linear_detrending": self.timesfm_use_linear_detrending,
            "linear_detrending_threshold": self.timesfm_linear_detrending_threshold,
            "use_iterative_cpm_revin": self.timesfm_use_iterative_cpm_revin,
            "use_frozen_running_stats": self.timesfm_use_frozen_running_stats,
            "use_variate_attention": self.timesfm_use_variate_attention,
            "use_sdpa": self.timesfm_use_sdpa,
            "value_clip": self.timesfm_value_clip,
            "input_transform": self.timesfm_input_transform,
            "cache_dir": self.timesfm_cache_dir,
            "revision": self.timesfm_revision,
            "local_files_only": self.timesfm_local_files_only,
            "token": self.hf_token,
        }


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()
