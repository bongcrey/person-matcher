"""
Unit and integration tests for person_matcher.
"""

from __future__ import annotations

import os
import unittest

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from person_matcher import RecordLinker
from person_matcher.utils import UnionFind, blocking_key_expr, load_file

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PERSONS_A = os.path.join(DATA_DIR, "persons_a.csv")
PERSONS_B = os.path.join(DATA_DIR, "persons_b.csv")
PERSONS_DUPES = os.path.join(DATA_DIR, "persons_dupes.csv")


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("PersonMatcherTests")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


class TestUnionFind(unittest.TestCase):
    def test_singleton(self):
        uf = UnionFind()
        assert uf.find("x") == "x"

    def test_union_merges_components(self):
        uf = UnionFind()
        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")

    def test_transitivity(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.find("a") == uf.find("c")

    def test_different_components(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("c", "d")
        assert uf.find("a") != uf.find("c")

    def test_assign_cluster_ids(self):
        uf = UnionFind()
        mapping = uf.assign_cluster_ids([("x", "y"), ("y", "z")])
        assert mapping["x"] == mapping["y"] == mapping["z"]

    def test_idempotent_union(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("a", "b")
        assert len(uf.components()) == 2


class TestBlockingKey:
    def test_prefix_length(self, spark):
        df = spark.range(1).withColumn("given_name", F.lit("  Alice  "))
        df = df.withColumn("_block", blocking_key_expr("given_name", prefix_len=2))
        assert df.collect()[0]["_block"] == "al"

    def test_handles_null(self, spark):
        df = spark.range(1).withColumn("given_name", F.lit(None).cast(StringType()))
        df = df.withColumn("_block", blocking_key_expr("given_name", prefix_len=2))
        assert df.collect()[0]["_block"] is None


class TestLoadFile:
    def test_csv_loads(self, spark):
        df = load_file(spark, PERSONS_A, id_col="rec_id")
        assert df.count() == 10
        assert "rec_id" in df.columns

    def test_unsupported_format_raises(self, spark):
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_file(spark, "data.json")


class TestPreprocess:
    def test_lowercase_trim(self, spark):
        df = (
            spark.range(1)
            .withColumn("given_name", F.lit("  JOHN  "))
            .withColumn("surname", F.lit("  SMITH  "))
            .withColumn("date_of_birth", F.lit("1985-03-12"))
            .withColumn("gender", F.lit("M"))
        )
        result = RecordLinker(spark).preprocess(df)
        row = result.collect()[0]
        assert row["given_name"] == "john"
        assert row["surname"] == "smith"
        assert row["gender"] == "m"

    def test_id_col_injected(self, spark):
        df = spark.createDataFrame([("john", "smith")], ["given_name", "surname"])
        result = RecordLinker(spark).preprocess(df, id_col="rec_id")
        assert "rec_id" in result.columns


class TestDedup:
    """Known pairs: d001↔d002, d003↔d004, d007↔d008"""

    EXPECTED_PAIRS = [("d001", "d002"), ("d003", "d004"), ("d007", "d008")]

    def _get_cluster_map(self, spark):
        df = load_file(spark, PERSONS_DUPES, id_col="rec_id")
        linker = RecordLinker(spark, blocking_key="given_name", match_threshold=2.0,
                              blocking_prefix_len=2, num_partitions=4)
        result = linker.compute_matches(df, id_col="rec_id", output_col="cluster_id")
        return {r["rec_id"]: r["cluster_id"] for r in result.select("rec_id", "cluster_id").collect()}

    def test_known_pairs_share_cluster(self, spark):
        m = self._get_cluster_map(spark)
        for a, b in self.EXPECTED_PAIRS:
            assert m[a] == m[b], f"Expected {a} and {b} to share cluster"

    def test_distinct_records_differ(self, spark):
        m = self._get_cluster_map(spark)
        assert m["d001"] != m["d005"]

    def test_all_records_annotated(self, spark):
        assert len(self._get_cluster_map(spark)) == 10


class TestLinking:
    """All a00X / b00X pairs represent the same person."""

    EXPECTED_PAIRS = [(f"a{i:03d}", f"b{i:03d}") for i in range(1, 11)]

    def _get_match_maps(self, spark):
        df_a = load_file(spark, PERSONS_A, id_col="rec_id")
        df_b = load_file(spark, PERSONS_B, id_col="rec_id")
        linker = RecordLinker(spark, blocking_key="given_name", match_threshold=2.0,
                              blocking_prefix_len=1, num_partitions=4)
        ra, rb = linker.compute_matches(df_a, df_b, id_col="rec_id", output_col="match_id")
        map_a = {r["rec_id"]: r["match_id"] for r in ra.select("rec_id", "match_id").collect()}
        map_b = {r["rec_id"]: r["match_id"] for r in rb.select("rec_id", "match_id").collect()}
        return map_a, map_b

    def test_matched_pairs_share_id(self, spark):
        map_a, map_b = self._get_match_maps(spark)
        for a_id, b_id in self.EXPECTED_PAIRS:
            assert map_a.get(a_id) is not None
            assert map_a.get(a_id) == map_b.get(b_id)

    def test_both_files_have_all_records(self, spark):
        map_a, map_b = self._get_match_maps(spark)
        assert len(map_a) == 10
        assert len(map_b) == 10


class TestGeneratePairs:
    def test_dedup_schema(self, spark):
        df = load_file(spark, PERSONS_DUPES, id_col="rec_id")
        pairs = RecordLinker(spark, match_threshold=1.5, num_partitions=2).generate_pairs(df, id_col="rec_id")
        assert set(pairs.columns) == {"id_a", "id_b", "score"}

    def test_scores_in_range(self, spark):
        df = load_file(spark, PERSONS_DUPES, id_col="rec_id")
        pairs = RecordLinker(spark, match_threshold=0.0, num_partitions=2).generate_pairs(df, id_col="rec_id")
        for r in pairs.collect():
            assert 0.0 <= r["score"] <= 4.0
