"""
Person Record Linkage – PySpark + recordlinkage integration.

Modes
-----
- **Deduplication**: detect duplicate rows within a single file.
- **Linking**: match rows across two separate files.

Typical usage::

    from pyspark.sql import SparkSession
    from person_matcher import RecordLinker

    spark = SparkSession.builder.getOrCreate()
    linker = RecordLinker(spark, blocking_key="surname", match_threshold=2.5)
    result = linker.compute_matches(df_a, df_b)   # linking
    result = linker.compute_matches(df_a)          # deduplication
"""

from .linker import RecordLinker

__all__ = ["RecordLinker"]
