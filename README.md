# Person Matcher

A command-line tool for record linkage and deduplication of person records using PySpark and machine learning techniques. This tool helps identify and link duplicate or matching person records across datasets based on similarity scores.

## Features

- **Deduplication**: Remove duplicate records from a single dataset by clustering similar records.
- **Linking**: Match records between two datasets to identify potential duplicates or related entries.
- **Blocking**: Use blocking keys (e.g., surname or given name prefixes) to partition data and improve matching efficiency.
- **Configurable Thresholds**: Set similarity thresholds to control the strictness of matching.
- **Output Formats**: Support for CSV and Parquet output formats.
- **Scalable**: Built on PySpark for handling large datasets.

## Installation

1. Ensure you have Python 3.8+ installed.
2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. (Optional) Set up a virtual environment:

   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```

## Usage

The tool provides two main commands: `dedup` for deduplication and `link` for linking.

### General Options

- `--input`: Path to the input CSV file (for dedup).
- `--input-a`, `--input-b`: Paths to the two input CSV files (for link).
- `--output`: Path to the output directory.
- `--blocking-key`: Column to use for blocking (e.g., surname).
- `--threshold`: Similarity threshold for matching (default varies).
- `--output-format`: Output format (csv or parquet, default: csv).
- `--id-col`: Column name for record IDs (default: id).
- `--output-col`: Column name for cluster/match IDs (default: cluster_id).
- `--blocking-prefix`: Length of blocking key prefix (default: 1).
- `--num-partitions`: Number of Spark partitions (default: 10).

### Deduplication

Remove duplicates from a single dataset:

```
python cli.py dedup --input persons.csv --output results/deduped/ --blocking-key surname --threshold 2.5
```

### Linking

Link records between two datasets:

```
python cli.py link --input-a persons_a.csv --input-b persons_b.csv --output results/linked/ --blocking-key given_name --threshold 2.0 --output-format parquet
```

## Requirements

- Python 3.8+
- PySpark
- recordlinkage
- pandas
- See `requirements.txt` for full list.

## How It Works

The tool uses the `recordlinkage` library to compute similarity scores between records based on string comparisons (e.g., Jaro-Winkler distance for names) and exact matches for fields like date of birth or gender. It employs a blocking strategy to reduce computational complexity by grouping records with similar blocking keys before comparison.

Matches are resolved into clusters using a Union-Find algorithm, assigning cluster IDs to grouped records.

## Contributing

Contributions are welcome. Please ensure code follows the existing style and includes tests.

## License

This project is licensed under the MIT License.