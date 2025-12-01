from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# Shared feature schema between offline dataset and online feature builder
FEATURE_COLS: List[str] = [
    "ret_1",
    "ret_log_1",
    "ret_mean_3",
    "ret_std_3",
    "ret_mean_5",
    "ret_std_5",
    "ret_mean_10",
    "ret_std_10",
    "vol_sum_3",
    "vol_sum_5",
    "vol_sum_10",
]


class DatasetBuilder:
    """
    Loads tick CSVs for a symbol, builds a feature matrix and binary targets.
    The tick schema is expected to be: timestamp,price,qty,side
    """

    def __init__(self, symbol: str = "BTCUSDT", horizon: int = 1, data_dir: Optional[Path] = None):
        self.root = Path(__file__).resolve().parents[3]
        self.symbol = symbol
        self.horizon = horizon
        self.data_dir = data_dir or (self.root / "data" / "ticks")
        self.fallback_dir = self.root / "data" / "offline"

    def _collect_files(self) -> List[Path]:
        patterns = [f"{self.symbol}_*.csv"]
        files: List[Path] = []
        for base in (self.data_dir, self.fallback_dir):
            if base.exists():
                for pattern in patterns:
                    files.extend(base.glob(pattern))
        return sorted({p.resolve() for p in files})

    def _load_ticks(self) -> pd.DataFrame:
        files = self._collect_files()
        if not files:
            print(f"[WARN] No tick files found for {self.symbol} in {self.data_dir} or {self.fallback_dir}")
            return pd.DataFrame()

        frames = []
        for fp in files:
            try:
                frames.append(pd.read_csv(fp))
            except Exception as exc:
                print(f"[WARN] Skipping {fp.name}: {exc}")
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = ["timestamp", "price", "qty", "side"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in ticks: {missing}")

        df = df.copy()
        df = df[required_cols + [c for c in df.columns if c not in required_cols]]
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
        df["side"] = df["side"].astype(str).str.lower()

        df = df.dropna(subset=["timestamp", "price", "qty"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        return df

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ret_1"] = df["price"].pct_change()
        df["ret_log_1"] = np.log(df["price"]).diff()

        for window in (3, 5, 10):
            df[f"ret_mean_{window}"] = df["ret_1"].rolling(window).mean()
            df[f"ret_std_{window}"] = df["ret_1"].rolling(window).std(ddof=0)
            df[f"vol_sum_{window}"] = df["qty"].rolling(window).sum()

        df["future_price"] = df["price"].shift(-self.horizon)
        df["target"] = (df["future_price"] > df["price"]).astype(int)

        df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
        return df

    def build(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        raw = self._load_ticks()
        if raw.empty:
            print(f"[ERROR] No data available for {self.symbol}. Ensure tick CSVs exist in {self.data_dir}.")
            return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()

        try:
            normalized = self._normalize_schema(raw)
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()

        featured = self._compute_features(normalized)
        if featured.empty:
            print(f"[ERROR] Not enough data to compute features/targets for {self.symbol}.")
            return pd.DataFrame(), pd.Series(dtype=int), normalized

        X = featured[FEATURE_COLS].copy()
        y = featured["target"].astype(int).copy()
        return X, y, featured
