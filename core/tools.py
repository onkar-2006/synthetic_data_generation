import pandas as pd
import json
import io
from langchain.tools import tool
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from pandas import DataFrame
from typing import Dict, Any


# --- Tool 1: Pandas Analysis (Used by Schema Inference Agent) ---
@tool
def analyze_dataframe_stats(df: DataFrame) -> str:
    """
    Analyzes a Pandas DataFrame (df) to extract key statistical information
    needed for synthetic data generation. Returns a JSON string summary.
    """
    summary = {}
    if df.empty: return "DataFrame is empty, cannot perform analysis."

    for column in df.columns:
        col = df[column]
        col_type = col.dtype
        # FIX: Cast nunique to standard Python int for JSON serialization
        stats = {'type': str(col_type), 'unique_count': int(col.nunique())}

        if pd.api.types.is_numeric_dtype(col_type):
            # FIX: Cast numpy types to standard Python types (float/int)
            stats.update({
                'mean': float(col.mean()),
                'std': float(col.std()),
                'min': col.min().item() if hasattr(col.min(), 'item') else col.min(),
                'max': col.max().item() if hasattr(col.max(), 'item') else col.max()
            })
        elif col.nunique() < 20 and not pd.api.types.is_numeric_dtype(col_type):
            # FIX: Ensure list items are standard types
            stats['top_values'] = col.value_counts().head(5).index.tolist()

        summary[column] = stats

    return json.dumps(summary, indent=2)


# --- Tool 2: SDV Synthesis (Used by Data Generation Agent) ---
@tool
def generate_synthetic_data_tool(original_data_csv: str, num_rows: int = 5000) -> str:
    """
    Generates synthetic data using the SDV library (GaussianCopula).
    Returns the synthetic data as a CSV string.
    """
    try:
        df = pd.read_csv(io.StringIO(original_data_csv))

        # FIX: Drop index column if it exists to prevent SDV errors
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        # FIX: Instantiate metadata object first (Required for SDV 1.0+)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)

        # Initialize, fit, and sample the synthesizer
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(df)

        synthetic_df = synthesizer.sample(num_rows=num_rows)

        return synthetic_df.to_csv(index=False)

    except Exception as e:
        return f"SYNTHESIS_ERROR: {e}"