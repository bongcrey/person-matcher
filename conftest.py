"""
Root conftest.py – apply environment fixes before PySpark initialises the JVM.

On Windows with Python 3.13:
- SPARK_HOME must be unset so PySpark uses its own bundled JARs instead of any
  system installation (e.g. an old Spark 2.x that lacks PythonUtils).
- PYSPARK_PYTHON / PYSPARK_DRIVER_PYTHON must point to the current interpreter
  so worker subprocesses resolve correctly instead of hitting the Windows Store stub.
"""

import os
import sys

os.environ.pop("SPARK_HOME", None)
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
