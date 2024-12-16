# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "httpx",
#   "chardet",
#   "python-dotenv",
# ]
# ///

import os
import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import httpx
import chardet
from dotenv import load_dotenv

# Force non-interactive matplotlib backend
matplotlib.use('Agg')

# Load environment variables
load_dotenv()

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    raise ValueError("API token not set. Please set AIPROXY_TOKEN in the environment.")

def detect_encoding(file_path):
    """Detect the file encoding."""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def load_data(file_path):
    """Load CSV data with detected encoding."""
    encoding = detect_encoding(file_path)
    try:
        return pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def analyze_data(df):
    """Perform basic data analysis."""
    numeric_df = df.select_dtypes(include=['number'])
    analysis = {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict()
    }
    return analysis

def visualize_data(df, output_dir):
    """Generate and save visualizations, including correlation heatmap and histograms for top numeric columns."""
    sns.set(style="whitegrid")
    numeric_df = df.select_dtypes(include=['number'])

    # Generate correlation heatmap if applicable
    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.title("Correlation Heatmap")
        plt.savefig(heatmap_path, bbox_inches='tight')
        plt.close()
        print(f"Saved correlation heatmap: {heatmap_path}")

    # Find top 2 numeric columns with highest variability
    if numeric_df.shape[1] > 0:
        numeric_variability = numeric_df.std().sort_values(ascending=False)
        top_columns = numeric_variability.head(2).index

        for column in top_columns:
            plt.figure()
            sns.histplot(numeric_df[column].dropna(), kde=True, bins=30, color="skyblue")
            plt.title(f"Distribution of {column}")
            hist_path = os.path.join(output_dir, f"{column}_histogram.png")
            plt.savefig(hist_path, bbox_inches='tight')
            plt.close()
            print(f"Saved histogram: {hist_path}")

def generate_narrative(analysis):
    """Generate narrative using LLM."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    prompt = (
        f"Provide a detailed analysis based on the following data summary:\n"
        f"Summary Statistics: {analysis['summary']}\n"
        f"Missing Values: {analysis['missing_values']}\n"
        f"Correlation Matrix: {analysis['correlation']}\n"
    )
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating narrative: {e}")
        return "Narrative generation failed."

def main():
    parser = argparse.ArgumentParser(description="Analyze datasets and generate insights.")
    parser.add_argument("file_path", help="Path to the dataset CSV file.")
    parser.add_argument("-o", "--output_dir", default="output", help="Directory to save outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = load_data(args.file_path)

    # Analyze data
    analysis = analyze_data(df)

    # Visualize data
    visualize_data(df, args.output_dir)

    # Generate narrative
    narrative = generate_narrative(analysis)

    # Save narrative
    narrative_path = os.path.join(args.output_dir, 'README.md')
    with open(narrative_path, 'w') as f:
        f.write(narrative)
    print(f"Saved narrative: {narrative_path}")

if __name__ == "__main__":
    main()