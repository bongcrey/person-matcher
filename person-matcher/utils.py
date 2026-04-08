"""
Utility helpers for person record linkage.

Includes:
- UnionFind: path-compressed union-find for cluster assignment.
- blocking_key_col: derive a Spark column expression for the blocking key.
- load_file: load CSV or Parquet into a Spark DataFrame.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set Union)
# ---------------------------------------------------------------------------

class UnionFind:
    """
    Path-compressed, union-by-rank disjoint set structure.

    Used to collapse transitive match pairs into clusters so that
    records {A↔B, B↔C} all share the same cluster_id.
    """

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def _ensure(self, x: str) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: str) -> str:
        """Return the canonical representative of *x*'s component."""
        self._ensure(x)
        # Path compression
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        """Merge the components containing *x* and *y*."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Union by rank
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def components(self) -> Dict[str, str]:
        """Return {element: root} mapping for every known element."""
        return {x: self.find(x) for x in self._parent}

    def assign_cluster_ids(self, pairs: Iterable[Tuple[str, str]]) -> Dict[str, str]:
        """
        Consume an iterable of (id_a, id_b) match pairs and return a
        ``{record_id: cluster_id}`` mapping where the cluster_id is the
        canonical root for that component.
        """
        for a, b in pairs:
            self.union(a, b)
        return self.components()


# ---------------------------------------------------------------------------
# Blocking-key column expression
# ---------------------------------------------------------------------------

def blocking_key_expr(col_name: str, prefix_len: int = 2) -> F.Column:
    """
    Return a Spark ``Column`` expression representing the blocking key.

    The blocking key is the first *prefix_len* characters of *col_name*
    (after lower-casing and trimming).  Records that share the same blocking
    key are placed in the same partition for pairwise comparison.

    Parameters
    ----------
    col_name:
        Name of the DataFrame column to derive the key from.
    prefix_len:
        Number of leading characters to use (default 2).
    """
    return F.substring(F.lower(F.trim(F.col(col_name))), 1, prefix_len).alias("_block")


# ---------------------------------------------------------------------------
# File loader
# ---------------------------------------------------------------------------

def load_file(spark: SparkSession, path: str, id_col: str = "rec_id") -> DataFrame:
    """
    Load a CSV or Parquet file into a Spark DataFrame.

    - CSV files: header is inferred automatically; schema is inferred.
    - Parquet files: schema is read from the file metadata.

    A stable ``id_col`` column (string) is added via
    ``monotonically_increasing_id`` if not already present.

    Parameters
    ----------
    spark:
        Active ``SparkSession``.
    path:
        Absolute or relative path to the file.
    id_col:
        Name of the record-identifier column to create (if absent).

    Returns
    -------
    pyspark.sql.DataFrame
    """
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".parquet":
        df = spark.read.parquet(path)
        logger.info("Loaded Parquet file: %s  (%d rows)", path, df.count())
    elif ext == ".csv":
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(path)
        logger.info("Loaded CSV file: %s  (%d rows)", path, df.count())
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported formats: .csv, .parquet"
        )

    if id_col not in df.columns:
        df = df.withColumn(id_col, F.monotonically_increasing_id().cast(StringType()))

    return df
