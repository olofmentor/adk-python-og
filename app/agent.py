from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DEFAULT_DATASET_PATH = "/app/data/dataset.csv"


@dataclass
class AgentConfig:
    dataset_path: str = DEFAULT_DATASET_PATH


class AnalysisAgent:
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig()
        self._df_cache: Optional[pd.DataFrame] = None

    def _load_df(self) -> pd.DataFrame:
        if self._df_cache is None:
            self._df_cache = pd.read_csv(self.config.dataset_path)
        return self._df_cache

    def _numeric_columns(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    def _parse_prompt(self, prompt: str) -> Dict[str, Any]:
        text = prompt.strip().lower()
        # Patterns:
        # 1) summary stats
        if re.search(r"\b(summary|summar(y|ise|ize)|describe)\b", text):
            return {"task": "summary"}
        # 2) correlation matrix
        if "correlation" in text or "corr" in text:
            return {"task": "correlation"}
        # 3) mean/avg of column optionally grouped by group
        m = re.search(r"(mean|average|avg) of ([a-zA-Z0-9_]+)(?: grouped by ([a-zA-Z0-9_]+))?", text)
        if m:
            return {"task": "mean", "column": m.group(2), "group": m.group(3)}
        # 4) linear regression: predict y from x1,x2
        m = re.search(r"regression: predict ([a-zA-Z0-9_]+) from ([a-zA-Z0-9_, ]+)", text)
        if m:
            target = m.group(1)
            features = [f.strip() for f in m.group(2).split(",")]
            return {"task": "linreg", "target": target, "features": features}
        # 5) filter and aggregate: mean col where condition (safe subset of expressions)
        m = re.search(r"mean of ([a-zA-Z0-9_]+) where ([a-zA-Z0-9_ <>=!&|.'\-]+)", text)
        if m:
            return {"task": "conditional_mean", "column": m.group(1), "condition": m.group(2)}
        # Fallback: pandas describe
        return {"task": "summary"}

    def _safe_query(self, df: pd.DataFrame, expr: str) -> pd.DataFrame:
        # Guard: allow only column names, numbers, operators and simple string literals
        if not re.fullmatch(r"[\w\s<>=!&|().'\-]+", expr):
            raise ValueError("Unsafe expression")
        # Ensure tokens correspond to columns or allowed keywords/operators
        allowed_tokens = set(df.columns) | set(["and", "or", "not"])
        # Tokenize by non-word
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)
        for tok in tokens:
            if tok.lower() not in allowed_tokens and not tok.isnumeric():
                # Allow common operators/func names as tokens check misses numbers already
                raise ValueError(f"Unknown token in expression: {tok}")
        return df.query(expr)

    def run(self, prompt: str) -> Dict[str, Any]:
        df = self._load_df()
        plan = self._parse_prompt(prompt)
        task = plan["task"]

        if task == "summary":
            desc = df.describe(include="all").to_dict()
            return {"type": "summary", "data": desc}

        if task == "correlation":
            numeric_df = df[self._numeric_columns(df)]
            corr = numeric_df.corr(numeric_only=True)
            return {"type": "correlation", "data": corr.to_dict()}

        if task == "mean":
            column = plan["column"]
            group = plan.get("group")
            if column not in df.columns:
                raise ValueError(f"Unknown column: {column}")
            if group:
                if group not in df.columns:
                    raise ValueError(f"Unknown group column: {group}")
                result = df.groupby(group)[column].mean(numeric_only=True)
                return {"type": "group_mean", "column": column, "group": group, "data": result.to_dict()}
            else:
                value = float(df[column].mean(numeric_only=True))
                return {"type": "mean", "column": column, "value": value}

        if task == "conditional_mean":
            column = plan["column"]
            condition = plan["condition"]
            if column not in df.columns:
                raise ValueError(f"Unknown column: {column}")
            filtered = self._safe_query(df, condition)
            value = float(filtered[column].mean(numeric_only=True))
            return {"type": "conditional_mean", "column": column, "condition": condition, "value": value}

        if task == "linreg":
            target = plan["target"]
            features = plan["features"]
            for c in [target] + features:
                if c not in df.columns:
                    raise ValueError(f"Unknown column: {c}")
            x = df[features].select_dtypes(include=[np.number]).dropna()
            y = df[[target]].loc[x.index].select_dtypes(include=[np.number]).dropna()
            x = x.loc[y.index]
            if len(x) == 0 or len(y) == 0:
                raise ValueError("Insufficient numeric data for regression")
            model = LinearRegression().fit(x.values, y.values.ravel())
            r2 = float(model.score(x.values, y.values.ravel()))
            coefs = {features[i]: float(model.coef_[i]) for i in range(len(features))}
            intercept = float(model.intercept_)
            return {"type": "linreg", "target": target, "features": features, "r2": r2, "coefficients": coefs, "intercept": intercept}

        raise ValueError("Unsupported task")