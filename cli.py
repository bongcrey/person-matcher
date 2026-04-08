#!/usr/bin/env python3
"""
CLI entry point for Person Record Linkage.

Usage examples
--------------
Deduplication::

    python cli.py dedup \\
        --input persons.csv \\
        --output results/deduped/ \\
        --blocking-key surname \\
        --threshold 2.5

Linking::

    python cli.py link \\
        --input-a persons_a.csv \\
        --input-b persons_b.csv \\
        --output results/linked/ \\
        --blocking-key given_name \\
        --threshold 2.0 \\
        --output-format parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
from typing import Optional

# Must be set before PySpark initialises the JVM so worker subprocesses
# use the same interpreter as the driver instead of the bare 'python' stub.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

# Remove any system-level SPARK_HOME that points to a different Spark
# installation; PySpark should use its own bundled JARs instead.
os.environ.pop("SPARK_HOME", None)

from pyspark.sql import SparkSession

from person_matcher import RecordLinker
from person_matcher.utils import load_file


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
        level=numeric,
        stream=sys.stderr,
    )
    for noisy in ("py4j", "pyspark"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _build_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .getOrCreate()
    )


def _write(df, path: str, fmt: str) -> None:
    os.makedirs(path, exist_ok=True)
    # Use pandas for output to avoid Hadoop native-IO issues on Windows
    # (Spark's CSV/Parquet committer requires hadoop.dll which may not be present).
    pdf = df.toPandas()
    out_file = os.path.join(path, f"part-00000.{fmt}")
    if fmt == "parquet":
        pdf.to_parquet(out_file, index=False)
    else:
        pdf.to_csv(out_file, index=False)
    logging.getLogger(__name__).info("Wrote %s output to: %s", fmt.upper(), out_file)


def cmd_dedup(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)
    spark = _build_spark("PersonMatcher-Dedup")
    df = load_file(spark, args.input, id_col=args.id_col)

    linker = RecordLinker(
        spark,
        blocking_key=args.blocking_key,
        match_threshold=args.threshold,
        blocking_prefix_len=args.blocking_prefix,
        num_partitions=args.num_partitions,
    )
    result = linker.compute_matches(df, id_col=args.id_col, output_col=args.output_col)
    _write(result, args.output, args.output_format)

    total = result.count()
    clusters = result.select(args.output_col).distinct().count()
    print(f"\n{'='*50}")
    print(f"  Mode          : deduplication")
    print(f"  Input records : {total}")
    print(f"  Clusters found: {clusters}")
    print(f"  Output path   : {args.output}")
    print(f"{'='*50}\n")


def cmd_link(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)
    spark = _build_spark("PersonMatcher-Link")
    df_a = load_file(spark, args.input_a, id_col=args.id_col)
    df_b = load_file(spark, args.input_b, id_col=args.id_col)

    linker = RecordLinker(
        spark,
        blocking_key=args.blocking_key,
        match_threshold=args.threshold,
        blocking_prefix_len=args.blocking_prefix,
        num_partitions=args.num_partitions,
    )
    result_a, result_b = linker.compute_matches(
        df_a, df_b, id_col=args.id_col, output_col=args.output_col
    )

    _write(result_a, f"{args.output}/file_a", args.output_format)
    _write(result_b, f"{args.output}/file_b", args.output_format)

    matched_a = result_a.filter(result_a[args.output_col].isNotNull()).count()
    matched_b = result_b.filter(result_b[args.output_col].isNotNull()).count()
    groups = result_a.select(args.output_col).filter(result_a[args.output_col].isNotNull()).distinct().count()

    print(f"\n{'='*50}")
    print(f"  Mode               : linking")
    print(f"  File-A records     : {result_a.count()}")
    print(f"  File-B records     : {result_b.count()}")
    print(f"  Matched in A       : {matched_a}")
    print(f"  Matched in B       : {matched_b}")
    print(f"  Match groups found : {groups}")
    print(f"  Output path        : {args.output}")
    print(f"{'='*50}\n")


def _shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--blocking-key", default="given_name", metavar="COL")
    parser.add_argument("--threshold", type=float, default=2.5, metavar="FLOAT")
    parser.add_argument("--blocking-prefix", type=int, default=2, metavar="N")
    parser.add_argument("--num-partitions", type=int, default=10, metavar="N")
    parser.add_argument("--output-format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--id-col", default="rec_id", metavar="COL")
    parser.add_argument("--output-col", default="cluster_id", metavar="COL")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="person-matcher",
        description="Person record linkage powered by PySpark + recordlinkage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dedup_p = sub.add_parser("dedup", help="Find duplicates within a single file.")
    dedup_p.add_argument("--input", required=True, metavar="PATH")
    dedup_p.add_argument("--output", required=True, metavar="PATH")
    _shared_args(dedup_p)

    link_p = sub.add_parser("link", help="Match records across two files.")
    link_p.add_argument("--input-a", required=True, metavar="PATH")
    link_p.add_argument("--input-b", required=True, metavar="PATH")
    link_p.add_argument("--output", required=True, metavar="PATH")
    _shared_args(link_p)

    return parser


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    if args.command == "dedup":
        cmd_dedup(args)
    elif args.command == "link":
        cmd_link(args)


if __name__ == "__main__":
    main()
