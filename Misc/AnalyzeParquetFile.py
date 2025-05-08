import pandas as pd
from pathlib import Path

def analyze_parquet_file(file_path):
    print(f"Loading parquet file from: {file_path}")
    try:
        df = pd.read_parquet(file_path)
        print(f"Successfully loaded data with shape: {df.shape}")

        results = {
            "file_path": str(file_path),
            "num_rows": df.shape[0],
            "num_columns": df.shape[1],
            "columns": list(df.columns),
            "column_info": {},
            "numerical_columns": [],
            "categorical_columns": [],
            "boolean_columns": [],
            "datetime_columns": [],
            "text_columns": [],
            "null_counts": {},
            "unique_values_count": {},
            "column_correlations": {}
        }

        null_counts = df.isnull().sum()
        for col in df.columns:
            results["null_counts"][col] = int(null_counts[col])

        for col in df.columns:
            col_data = df[col].dropna()
            col_type = str(df[col].dtype)
            unique_count = df[col].nunique()

            is_numerical = False
            if len(col_data) > 0:
                try:
                    sample = col_data.head(100)
                    sample.astype(float)
                    is_numerical = True
                except (ValueError, TypeError):
                    is_numerical = False

            if is_numerical:
                if set(col_data.unique()) == {0, 1} or col_data.dtype == bool:
                    col_category = "boolean"
                    results["boolean_columns"].append(col)
                else:
                    col_category = "numerical"
                    results["numerical_columns"].append(col)

                if col_category == "numerical":
                    results["column_info"][col] = {
                        "type": col_type,
                        "category": col_category,
                        "null_count": int(null_counts[col]),
                        "null_percentage": round(null_counts[col] / len(df) * 100, 2),
                        "unique_count": int(unique_count),
                        "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                        "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                        "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                        "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                        "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                    }
                else:
                    results["column_info"][col] = {
                        "type": col_type,
                        "category": col_category,
                        "null_count": int(null_counts[col]),
                        "null_percentage": round(null_counts[col] / len(df) * 100, 2),
                        "unique_count": int(unique_count),
                        "true_percentage": round(col_data.mean() * 100, 2) if len(col_data) > 0 else None,
                    }

            elif pd.api.types.is_datetime64_dtype(df[col]):
                col_category = "datetime"
                results["datetime_columns"].append(col)
                results["column_info"][col] = {
                    "type": col_type,
                    "category": col_category,
                    "null_count": int(null_counts[col]),
                    "null_percentage": round(null_counts[col] / len(df) * 100, 2),
                    "unique_count": int(unique_count),
                    "min": str(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": str(df[col].max()) if not pd.isna(df[col].max()) else None,
                }

            else:
                if unique_count < 0.2 * len(df) or unique_count < 50:
                    col_category = "categorical"
                    results["categorical_columns"].append(col)
                    value_counts = df[col].value_counts(normalize=True).head(10).to_dict()
                    value_counts = {str(k): float(v) for k, v in value_counts.items()}

                    results["column_info"][col] = {
                        "type": col_type,
                        "category": col_category,
                        "null_count": int(null_counts[col]),
                        "null_percentage": round(null_counts[col] / len(df) * 100, 2),
                        "unique_count": int(unique_count),
                        "top_categories": value_counts
                    }
                else:
                    col_category = "text"
                    results["text_columns"].append(col)
                    results["column_info"][col] = {
                        "type": col_type,
                        "category": "text",
                        "null_count": int(null_counts[col]),
                        "null_percentage": round(null_counts[col] / len(df) * 100, 2),
                        "unique_count": int(unique_count),
                    }

            results["unique_values_count"][col] = int(unique_count)

        if len(results["numerical_columns"]) > 1:
            corr_matrix = df[results["numerical_columns"]].corr()
            for col1 in corr_matrix.columns:
                results["column_correlations"][col1] = {}
                for col2 in corr_matrix.columns:
                    if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > 0.5:
                        results["column_correlations"][col1][col2] = round(float(corr_matrix.loc[col1, col2]), 3)

        sample_df = df.head(5)
        results["sample_data"] = sample_df.to_dict(orient='records')

        return results
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        return {"error": str(e)}

def main():
    file_path = Path("../Parquets/parsed_showdown_replays.parquet")

    if not file_path.exists():
        print(f"Error: File {file_path} does not exist!")
        return

    analysis_results = analyze_parquet_file(file_path)

    print("\nAnalysis Summary:")
    print(f"Number of rows: {analysis_results.get('num_rows', 'N/A')}")
    print(f"Number of columns: {analysis_results.get('num_columns', 'N/A')}")
    print(f"Columns in dataset: {analysis_results.get('columns', [])}")
    print("\nColumn Type Counts:")
    print(f"Numerical columns: {len(analysis_results.get('numerical_columns', []))}")
    print(f"Categorical columns: {len(analysis_results.get('categorical_columns', []))}")
    print(f"Boolean columns: {len(analysis_results.get('boolean_columns', []))}")
    print(f"Datetime columns: {len(analysis_results.get('datetime_columns', []))}")
    print(f"Text columns: {len(analysis_results.get('text_columns', []))}")

    print("\nColumn Categories:")
    print(f"Numerical columns: {analysis_results.get('numerical_columns', [])}")
    print(f"Categorical columns: {analysis_results.get('categorical_columns', [])}")
    print(f"Boolean columns: {analysis_results.get('boolean_columns', [])}")
    print(f"Datetime columns: {analysis_results.get('datetime_columns', [])}")
    print(f"Text columns: {analysis_results.get('text_columns', [])}")

    print("\nNull Values Per Column:")
    null_counts = analysis_results.get('null_counts', {})
    for col, count in null_counts.items():
        print(f"  {col}: {count} nulls ({round(count/analysis_results['num_rows']*100, 2)}%)")

    print("\nColumn Statistics:")
    for col, info in analysis_results.get('column_info', {}).items():
        print(f"  {col} ({info.get('category', 'unknown')}):")
        for key, value in info.items():
            if key not in ['category', 'type']:
                print(f"    {key}: {value}")

    print("\nStrong Correlations:")
    correlations = analysis_results.get('column_correlations', {})
    for col1, corr_dict in correlations.items():
        if corr_dict:
            print(f"  {col1} correlates with:")
            for col2, corr_val in corr_dict.items():
                print(f"    {col2}: {corr_val}")

    print("\nSample Data (First 5 rows):")
    for i, row in enumerate(analysis_results.get('sample_data', [])[:5]):
        print(f"  Row {i+1}:")
        for col, val in row.items():
            print(f"    {col}: {val}")

if __name__ == "__main__":
    main()