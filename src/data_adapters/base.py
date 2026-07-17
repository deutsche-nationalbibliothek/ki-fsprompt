from abc import ABC, abstractmethod
import logging

import pandas as pd

from .constants import REQUIRED_COLUMNS
from .types import AdapterOptions


class BaseDataAdapter(ABC):
    """Base contract for loading heterogeneous sources into one schema."""

    required_columns = REQUIRED_COLUMNS

    def __init__(
        self,
        *,
        options: AdapterOptions | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.options = options or AdapterOptions()
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def load(self) -> pd.DataFrame:
        self._validate_inputs()
        raw_data = self._load_raw()
        normalized = self._normalize(raw_data)
        normalized = self._validate_schema(normalized)
        normalized = self._apply_limits(normalized)
        self.logger.info("Adapter output shape=%s", normalized.shape)
        return normalized

    @abstractmethod
    def _validate_inputs(self) -> None:
        """Validate source-specific inputs before reading files."""

    @abstractmethod
    def _load_raw(self) -> pd.DataFrame:
        """Load source-specific raw data into an intermediate table."""

    @abstractmethod
    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize raw data to the required adapter schema."""

    def _validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Adapter output missing required columns: {missing_columns}"
            )

        if self.options.strict:
            null_mask = df[list(self.required_columns)].isnull().any()
            null_columns = [col for col, has_null in null_mask.items() if bool(has_null)]
            if null_columns:
                raise ValueError(
                    f"Adapter output has null values in required columns: {null_columns}"
                )

        return df

    def _apply_limits(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.options.max_rows is not None:
            df = df.head(self.options.max_rows)
            self.logger.info("Applied max_rows=%s", self.options.max_rows)

        if self.options.debug_rows is not None:
            df = df.head(self.options.debug_rows)
            self.logger.info("Applied debug_rows=%s", self.options.debug_rows)

        return df
