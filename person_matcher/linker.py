"""
Core RecordLinker class.

Architecture
------------
1. ``preprocess``  – normalise strings, ensure a stable record-id column.
2. ``generate_pairs`` – add a blocking-key column, repartition, then use
   ``mapInPandas`` to emit candidate (id_a, id_b, score) rows from within
   each block.
3. ``compute_matches`` – full pipeline: preprocess → generate_pairs →
   Union-Find cluster assignment → join cluster_id back to the source data.

Distributed strategy
--------------------
Data is repartitioned by a *blocking key* (prefix of ``blocking_key`` column).
Within each Spark partition a plain Python function (``_dedup_fn`` or
``_link_fn``) is invoked via ``mapInPandas``.  That function uses the
``recordlinkage`` library to:

* build a full-index over the records in the partition,
* compute comparison vectors (Jaro-Winkler for names, exact for dob/gender),
* filter pairs whose aggregate score meets ``match_threshold``.

The collected match pairs are then resolved into clusters via a
``UnionFind`` structure, and the resulting ``cluster_id`` (or ``match_id``)
is joined back to the original Spark DataFrames.
"""

from __future__ import annotations

import csv as _csv_mod
import logging
import os
import tempfile
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)

import recordlinkage as rl

from .utils import UnionFind, blocking_key_expr

logger = logging.getLogger(__name__)

# Schema returned by every mapInPandas comparator function
_PAIRS_SCHEMA = StructType(
    [
        StructField("id_a", StringType(), nullable=False),
        StructField("id_b", StringType(), nullable=False),
        StructField("score", DoubleType(), nullable=False),
    ]
)

# Columns that receive fuzzy-string comparison (Jaro-Winkler)
_FUZZY_COLS: List[str] = ["given_name", "surname"]

# Columns that receive exact comparison when present
_EXACT_COLS: List[str] = ["date_of_birth", "gender"]


# ---------------------------------------------------------------------------
# Internal helper – build a recordlinkage Compare object
# ---------------------------------------------------------------------------

def _build_compare(available: List[str]) -> rl.Compare:
    """
    Return a ``recordlinkage.Compare`` configured for the available columns.

    Scoring weights (max total = 4.0 with all four columns present):

    ============== ======= ============
    column         method  max score
    ============== ======= ============
    given_name     JW      1.0
    surname        JW      1.0
    date_of_birth  exact   1.0
    gender         exact   1.0
    ============== ======= ============
    """
    compare = rl.Compare()
    for col in _FUZZY_COLS:
        if col in available:
            compare.string(col, col, method="jarowinkler", label=f"s_{col}")
    for col in _EXACT_COLS:
        if col in available:
            compare.exact(col, col, label=f"s_{col}")
    return compare


# ---------------------------------------------------------------------------
# Partition-level comparison functions (used with mapInPandas)
# ---------------------------------------------------------------------------

def _dedup_fn(
    threshold: float, id_col: str
) -> "Callable[[Iterator[pd.DataFrame]], Iterator[pd.DataFrame]]":
    """
    Return a ``mapInPandas``-compatible function for *deduplication*.

    Each incoming pandas partition contains records that share the same
    blocking-key prefix.  The function performs a full pairwise comparison
    within the block and yields (id_a, id_b, score) rows for pairs that
    meet *threshold*.
    """

    def _fn(partition_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        empty = pd.DataFrame({"id_a": pd.Series(dtype=str),
                               "id_b": pd.Series(dtype=str),
                               "score": pd.Series(dtype=float)})
        for pdf in partition_iter:
            pdf = pdf.drop(columns=["_block", "_source"], errors="ignore")
            if len(pdf) < 2:
                yield empty
                continue

            pdf = pdf.reset_index(drop=True)
            pdf.index = pdf[id_col].astype(str)
            pdf.index.name = id_col

            available = list(pdf.columns)
            compare = _build_compare(available)

            indexer = rl.Index()
            indexer.full()
            try:
                pairs = indexer.index(pdf)
            except Exception:  # noqa: BLE001
                yield empty
                continue

            if len(pairs) == 0:
                yield empty
                continue

            vectors = compare.compute(pairs, pdf)
            vectors["score"] = vectors.sum(axis=1)
            matches = vectors[vectors["score"] >= threshold]

            if matches.empty:
                yield empty
                continue

            result = pd.DataFrame(
                {
                    "id_a": matches.index.get_level_values(0).astype(str),
                    "id_b": matches.index.get_level_values(1).astype(str),
                    "score": matches["score"].values.astype(float),
                }
            )
            yield result

    return _fn


def _link_fn(
    threshold: float, id_col_a: str, id_col_b: str
) -> "Callable[[Iterator[pd.DataFrame]], Iterator[pd.DataFrame]]":
    """
    Return a ``mapInPandas``-compatible function for *record linking*.

    Each incoming pandas partition contains rows from **both** datasets
    (distinguished by the ``_source`` column: ``'a'`` or ``'b'``).
    The function performs cross-file comparison within the block.
    """

    def _fn(partition_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        empty = pd.DataFrame({"id_a": pd.Series(dtype=str),
                               "id_b": pd.Series(dtype=str),
                               "score": pd.Series(dtype=float)})
        for pdf in partition_iter:
            pdf_a = pdf[pdf["_source"] == "a"].drop(columns=["_block", "_source"], errors="ignore").copy()
            pdf_b = pdf[pdf["_source"] == "b"].drop(columns=["_block", "_source"], errors="ignore").copy()

            if pdf_a.empty or pdf_b.empty:
                yield empty
                continue

            pdf_a = pdf_a.reset_index(drop=True)
            pdf_b = pdf_b.reset_index(drop=True)
            pdf_a.index = pdf_a[id_col_a].astype(str)
            pdf_a.index.name = id_col_a
            pdf_b.index = pdf_b[id_col_b].astype(str)
            pdf_b.index.name = id_col_b

            shared = [c for c in pdf_a.columns if c in pdf_b.columns and c != id_col_a and c != id_col_b]
            available = shared
            if not available:
                yield empty
                continue

            compare = _build_compare(available)

            indexer = rl.Index()
            indexer.full()
            try:
                pairs = indexer.index(pdf_a, pdf_b)
            except Exception:  # noqa: BLE001
                yield empty
                continue

            if len(pairs) == 0:
                yield empty
                continue

            vectors = compare.compute(pairs, pdf_a, pdf_b)
            vectors["score"] = vectors.sum(axis=1)
            matches = vectors[vectors["score"] >= threshold]

            if matches.empty:
                yield empty
                continue

            result = pd.DataFrame(
                {
                    "id_a": matches.index.get_level_values(0).astype(str),
                    "id_b": matches.index.get_level_values(1).astype(str),
                    "score": matches["score"].values.astype(float),
                }
            )
            yield result

    return _fn


# ---------------------------------------------------------------------------
# RecordLinker
# ---------------------------------------------------------------------------

class RecordLinker:
    """
    Distributed person record linkage using PySpark and the *recordlinkage*
    library.

    Parameters
    ----------
    spark:
        Active ``SparkSession``.
    blocking_key:
        Column name used to partition records into blocks.  Only records
        sharing the same key prefix are compared, dramatically reducing the
        O(n²) candidate space.  Defaults to ``"given_name"``.
    match_threshold:
        Minimum aggregate comparison score required to classify a pair as a
        match.  Scores are summed across all configured comparison columns
        (max ≈ 4.0 when all four standard columns are present).
        Defaults to ``2.5``.
    blocking_prefix_len:
        Number of leading characters of ``blocking_key`` to use as the
        partition key.  Shorter prefixes → larger blocks → higher recall but
        more computation.  Defaults to ``2``.
    num_partitions:
        Number of Spark partitions to repartition into before comparison.
        Tune this to match the number of executors × cores available.
        Defaults to ``10``.
    """

    def __init__(
        self,
        spark: SparkSession,
        blocking_key: str = "given_name",
        match_threshold: float = 2.5,
        blocking_prefix_len: int = 2,
        num_partitions: int = 10,
    ) -> None:
        self.spark = spark
        self.blocking_key = blocking_key
        self.match_threshold = match_threshold
        self.blocking_prefix_len = blocking_prefix_len
        self.num_partitions = num_partitions

    def preprocess(self, df: DataFrame, id_col: str = "rec_id") -> DataFrame:
        """
        Normalise a Spark DataFrame for record linkage.

        Steps:
        1. Add a stable string record-identifier column *id_col* if absent.
        2. Lower-case and trim all ``StringType`` columns.
        3. Cast ``date_of_birth`` to ``StringType`` for consistent comparison.
        """
        if id_col not in df.columns:
            df = df.withColumn(
                id_col, F.monotonically_increasing_id().cast(StringType())
            )

        str_cols = [
            f.name
            for f in df.schema.fields
            if isinstance(f.dataType, StringType) and f.name != id_col
        ]
        for col in str_cols:
            df = df.withColumn(col, F.lower(F.trim(F.col(col))))

        if "date_of_birth" in df.columns:
            df = df.withColumn("date_of_birth", F.col("date_of_birth").cast(StringType()))

        return df

    def generate_pairs(
        self,
        df_a: DataFrame,
        df_b: Optional[DataFrame] = None,
        id_col: str = "rec_id",
    ) -> DataFrame:
        """
        Produce candidate match pairs using the configured blocking strategy.

        Returns a DataFrame with schema ``(id_a STRING, id_b STRING, score DOUBLE)``.
        """
        if df_b is None:
            return self._generate_dedup_pairs(df_a, id_col=id_col)
        return self._generate_link_pairs(df_a, df_b, id_col_a=id_col, id_col_b=id_col)

    def compute_matches(
        self,
        df_a: DataFrame,
        df_b: Optional[DataFrame] = None,
        id_col: str = "rec_id",
        output_col: str = "cluster_id",
    ) -> DataFrame | Tuple[DataFrame, DataFrame]:
        """
        Full pipeline: preprocess → block → compare → classify → annotate.

        Deduplication mode: returns df_a with ``cluster_id`` column added.
        Linking mode: returns ``(result_a, result_b)`` tuple with ``match_id`` added.
        """
        logger.info(
            "compute_matches: mode=%s, blocking_key=%s, threshold=%.2f",
            "deduplication" if df_b is None else "linking",
            self.blocking_key,
            self.match_threshold,
        )

        df_a = self.preprocess(df_a, id_col=id_col)

        if df_b is None:
            return self._run_deduplication(df_a, id_col=id_col, output_col=output_col)

        df_b = self.preprocess(df_b, id_col=id_col)
        return self._run_linking(df_a, df_b, id_col=id_col, output_col=output_col)

    # ------------------------------------------------------------------
    # Internal – deduplication
    # ------------------------------------------------------------------

    def _dict_to_sdf(self, d: Dict[str, str], key_col: str, val_col: str) -> DataFrame:
        """
        Persist *d* as a temp CSV and return a cached Spark DataFrame.

        ``createDataFrame(python_data)`` is avoided because PySpark 4.x spawns a
        Python worker subprocess even for small lists, which crashes on Windows
        with Python 3.13.  Reading from a local CSV file is handled entirely by
        the JVM and is safe on all platforms.
        """
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
        )
        tmp_path = tmp.name
        try:
            writer = _csv_mod.writer(tmp)
            writer.writerow([key_col, val_col])
            writer.writerows(d.items())
            tmp.close()
            sdf = self.spark.read.option("header", "true").csv(
                tmp_path.replace("\\", "/")
            )
            sdf = sdf.cache()
            sdf.count()  # materialise so we can safely delete the temp file
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return sdf

    def _pairs_to_sdf(self, pairs_pdf: pd.DataFrame) -> DataFrame:
        """Write (id_a, id_b, score) pandas DataFrame to temp CSV and return a cached Spark DataFrame."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
        )
        tmp_path = tmp.name
        try:
            pairs_pdf[["id_a", "id_b", "score"]].to_csv(tmp, index=False)
            tmp.close()
            sdf = self.spark.read.option("header", "true").csv(tmp_path.replace("\\", "/"))
            sdf = sdf.withColumn("score", F.col("score").cast(DoubleType())).cache()
            sdf.count()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return sdf

    def _generate_dedup_pairs(self, df: DataFrame, id_col: str = "rec_id") -> DataFrame:
        block_expr = blocking_key_expr(self.blocking_key, self.blocking_prefix_len)
        blocked_pdf = df.withColumn("_block", block_expr).toPandas()
        comparator = _dedup_fn(self.match_threshold, id_col)
        results: List[pd.DataFrame] = []
        for _, group in blocked_pdf.groupby("_block"):
            for chunk in comparator(iter([group])):
                if not chunk.empty:
                    results.append(chunk)
        pairs_pdf = pd.concat(results, ignore_index=True) if results else \
            pd.DataFrame({"id_a": pd.Series(dtype=str), "id_b": pd.Series(dtype=str), "score": pd.Series(dtype=float)})
        return self._pairs_to_sdf(pairs_pdf)

    def _run_deduplication(self, df: DataFrame, id_col: str, output_col: str) -> DataFrame:
        # Collect to pandas and compare locally – avoids mapInPandas Python workers
        block_expr = blocking_key_expr(self.blocking_key, self.blocking_prefix_len)
        blocked_pdf = df.withColumn("_block", block_expr).toPandas()
        comparator = _dedup_fn(self.match_threshold, id_col)
        pair_rows: List[Tuple[str, str]] = []
        for _, group in blocked_pdf.groupby("_block"):
            for chunk in comparator(iter([group])):
                if not chunk.empty:
                    pair_rows.extend(zip(chunk["id_a"].tolist(), chunk["id_b"].tolist()))
        logger.info("Dedup: %d candidate pairs above threshold", len(pair_rows))

        uf = UnionFind()
        all_ids = [str(r[id_col]) for r in df.select(id_col).collect()]
        for rid in all_ids:
            uf.find(rid)
        cluster_map: Dict[str, str] = uf.assign_cluster_ids(pair_rows)

        cluster_sdf = self._dict_to_sdf(cluster_map, id_col, output_col)
        result = df.join(cluster_sdf, on=id_col, how="left").cache()
        result.count()  # materialise before cluster_sdf temp file is gone
        logger.info("Dedup complete. Distinct clusters: %d", result.select(output_col).distinct().count())
        return result

    # ------------------------------------------------------------------
    # Internal – linking
    # ------------------------------------------------------------------

    def _generate_link_pairs(
        self, df_a: DataFrame, df_b: DataFrame, id_col_a: str = "rec_id", id_col_b: str = "rec_id"
    ) -> DataFrame:
        block_expr = blocking_key_expr(self.blocking_key, self.blocking_prefix_len)

        tagged_a = df_a.withColumn("_source", F.lit("a")).withColumn("_block", block_expr)
        tagged_b = df_b.withColumn("_source", F.lit("b")).withColumn("_block", block_expr)

        cols_a = set(tagged_a.columns)
        cols_b = set(tagged_b.columns)
        for c in cols_b - cols_a:
            tagged_a = tagged_a.withColumn(c, F.lit(None).cast(StringType()))
        for c in cols_a - cols_b:
            tagged_b = tagged_b.withColumn(c, F.lit(None).cast(StringType()))

        if id_col_a == id_col_b:
            tagged_b = tagged_b.withColumnRenamed(id_col_b, f"{id_col_b}_b")
            id_col_b_internal = f"{id_col_b}_b"
        else:
            id_col_b_internal = id_col_b

        all_cols = sorted(set(tagged_a.columns) | set(tagged_b.columns))
        for c in all_cols:
            if c not in tagged_a.columns:
                tagged_a = tagged_a.withColumn(c, F.lit(None).cast(StringType()))
            if c not in tagged_b.columns:
                tagged_b = tagged_b.withColumn(c, F.lit(None).cast(StringType()))
        tagged_a = tagged_a.select(all_cols)
        tagged_b = tagged_b.select(all_cols)

        combined_pdf = tagged_a.union(tagged_b).toPandas()
        comparator = _link_fn(self.match_threshold, id_col_a, id_col_b_internal)
        results: List[pd.DataFrame] = []
        for _, group in combined_pdf.groupby("_block"):
            for chunk in comparator(iter([group])):
                if not chunk.empty:
                    results.append(chunk)
        pairs_pdf = pd.concat(results, ignore_index=True) if results else \
            pd.DataFrame({"id_a": pd.Series(dtype=str), "id_b": pd.Series(dtype=str), "score": pd.Series(dtype=float)})
        return self._pairs_to_sdf(pairs_pdf)

    def _run_linking(
        self, df_a: DataFrame, df_b: DataFrame, id_col: str, output_col: str
    ) -> Tuple[DataFrame, DataFrame]:
        # Collect to pandas and compare locally – avoids mapInPandas Python workers
        block_expr = blocking_key_expr(self.blocking_key, self.blocking_prefix_len)
        tagged_a = df_a.withColumn("_source", F.lit("a")).withColumn("_block", block_expr)
        tagged_b = df_b.withColumn("_source", F.lit("b")).withColumn("_block", block_expr)

        cols_a = set(tagged_a.columns)
        cols_b = set(tagged_b.columns)
        for c in cols_b - cols_a:
            tagged_a = tagged_a.withColumn(c, F.lit(None).cast(StringType()))
        for c in cols_a - cols_b:
            tagged_b = tagged_b.withColumn(c, F.lit(None).cast(StringType()))

        if id_col == id_col:  # id_col_a == id_col_b in this call path
            tagged_b = tagged_b.withColumnRenamed(id_col, f"{id_col}_b")
            id_col_b_internal = f"{id_col}_b"
        else:
            id_col_b_internal = id_col

        all_cols = sorted(set(tagged_a.columns) | set(tagged_b.columns))
        for c in all_cols:
            if c not in tagged_a.columns:
                tagged_a = tagged_a.withColumn(c, F.lit(None).cast(StringType()))
            if c not in tagged_b.columns:
                tagged_b = tagged_b.withColumn(c, F.lit(None).cast(StringType()))
        combined_pdf = tagged_a.select(all_cols).union(tagged_b.select(all_cols)).toPandas()

        comparator = _link_fn(self.match_threshold, id_col, id_col_b_internal)
        pair_rows: List[Tuple[str, str]] = []
        for _, group in combined_pdf.groupby("_block"):
            for chunk in comparator(iter([group])):
                if not chunk.empty:
                    pair_rows.extend(zip(chunk["id_a"].tolist(), chunk["id_b"].tolist()))
        logger.info("Link: %d candidate pairs above threshold", len(pair_rows))

        uf = UnionFind()
        for a_id, b_id in pair_rows:
            uf.union(a_id, b_id)
        cluster_map = uf.components()

        match_map_a: Dict[str, str] = {}
        match_map_b: Dict[str, str] = {}
        for a_id, b_id in pair_rows:
            root = cluster_map[a_id]
            match_map_a[a_id] = root
            match_map_b[b_id] = root

        sdf_a = self._dict_to_sdf(match_map_a, id_col, output_col)
        sdf_b = self._dict_to_sdf(match_map_b, id_col, output_col)
        result_a = df_a.join(sdf_a, on=id_col, how="left").cache()
        result_b = df_b.join(sdf_b, on=id_col, how="left").cache()
        result_a.count()
        result_b.count()
        logger.info("Linking complete. %d match groups found.", len(set(match_map_a.values())))
        return result_a, result_b
