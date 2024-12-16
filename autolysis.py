import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
from datetime import datetime
from scipy import stats
import argparse

# Load the .env file from the specified directory
def load_env(file_name):
    dotenv_path = os.path.join("C:/Users/91735/OneDrive/Desktop/Analysis", file_name)
    load_dotenv(dotenv_path)
    return os.environ.get("AIPROXY_TOKEN")

# Initialize OpenAI API
def init_openai_api():
    api_token = load_env("myfile.env")
    if not api_token:
        raise ValueError("Error: AIPROXY_TOKEN not found in .env file.")
    return {
        "base_url": "https://aiproxy.sanand.workers.dev/openai/v1",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"
        }
    }

# Read CSV with fallback encoding
def read_csv(filename):
    try:
        return pd.read_csv(filename, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(filename, encoding="latin1")
    except Exception as e:
        raise ValueError(f"Error loading {filename}: {e}")

# Perform detailed statistical analysis on the dataset
def analyze_data(df):
    numeric_data = df.select_dtypes(include=[np.number])
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_statistics": df.describe(include="all").to_dict(),
        "skewness": numeric_data.skew().to_dict(),
        "kurtosis": numeric_data.kurt().to_dict(),
        "correlation_matrix": numeric_data.corr().to_dict() if not numeric_data.empty else {}
    }

# Identify outliers using the Interquartile Range (IQR) method
def identify_outliers(df):
    numeric_data = df.select_dtypes(include=[np.number])
    q1 = numeric_data.quantile(0.25)
    q3 = numeric_data.quantile(0.75)
    iqr = q3 - q1
    return ((numeric_data < (q1 - 1.5 * iqr)) | (numeric_data > (q3 + 1.5 * iqr))).sum().to_dict()

# Generate visualizations and save as images
def generate_visualizations(df, analysis, output_prefix):
    charts = []
    numeric_columns = df.select_dtypes(include=["number"]).columns
    if numeric_columns.empty:
        return charts

    # Correlation Heatmap
    if analysis.get("correlation_matrix"):
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        heatmap_file = f"{output_prefix}_heatmap.png"
        plt.title("Correlation Heatmap")
        plt.savefig(heatmap_file, dpi=100, bbox_inches='tight')
        charts.append(heatmap_file)
        plt.close()

    # Histograms
    for col in numeric_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True, bins=30, color="blue")
        plt.title(f"Distribution of {col}")
        hist_file = f"{output_prefix}_{col}_hist.png"
        plt.savefig(hist_file, dpi=100, bbox_inches='tight')
        charts.append(hist_file)
        plt.close()

    # Outlier Bar Plot
    outliers = identify_outliers(df)
    if outliers and sum(outliers.values()) > 0:
        plt.figure(figsize=(10, 6))
        pd.Series(outliers).plot(kind="bar", color="red")
        plt.title("Outlier Counts by Column")
        outliers_file = f"{output_prefix}_outliers.png"
        plt.savefig(outliers_file, dpi=100, bbox_inches='tight')
        charts.append(outliers_file)
        plt.close()

    return charts

# Construct a dynamic prompt for the LLM
def construct_prompt(analysis, charts, filename):
    chart_descriptions = "\n".join([f"- {chart}" for chart in charts])
    return f"""
    I analyzed a dataset from {filename}. Here are the details:
    - Shape: {analysis['shape']}
    - Columns: {analysis['columns']}
    - Missing Values: {analysis['missing_values']}
    - Summary Statistics: {analysis['summary_statistics']}
    - Skewness: {analysis['skewness']}
    - Kurtosis: {analysis['kurtosis']}

    The following charts were generated:
    {chart_descriptions}

    Write a detailed report summarizing the dataset, the analyses conducted, key insights derived, and actionable recommendations based on these insights.
    """

# Generate a detailed story from the LLM
def narrate_story(api, analysis, charts, filename):
    prompt = construct_prompt(analysis, charts, filename)
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        response = requests.post(
            f"{api['base_url']}/chat/completions", 
            headers=api['headers'], 
            json=data
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Story generation failed: {e}"

# Save story and visualizations to a markdown file
def save_readme(story, charts, output_file):
    try:
        with open(output_file, "w") as f:
            f.write("# Analysis Report\n\n")
            f.write(story + "\n\n")
            for chart in charts:
                f.write(f"![Chart](./{chart})\n")
        print(f"README saved to {output_file}")
    except Exception as e:
        print(f"Error saving README: {e}")

# Generate unique filename
def generate_unique_filename(filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename, extension = os.path.splitext(filename)
    return f"{filename}_{timestamp}{extension}"

# Main script execution
def main():
    parser = argparse.ArgumentParser(description="Automated Data Analysis Script")
    parser.add_argument("input", type=str, help="CSV file or folder containing the CSV files")
    
    args = parser.parse_args()
    
    # Check if input is a file or folder
    if os.path.isfile(args.input):
        csv_files = [os.path.basename(args.input)]
        os.chdir(os.path.dirname(args.input) or os.getcwd())
    elif os.path.isdir(args.input):
        os.chdir(args.input)
        csv_files = [f for f in os.listdir(args.input) if f.endswith('.csv')]
    else:
        print(f"Error: The input '{args.input}' is neither a valid file nor a folder.")
        return
    
    api = init_openai_api()

    for filename in csv_files:
        filepath = os.path.join(os.getcwd(), filename)
        df = read_csv(filepath)
        analysis = analyze_data(df)
        output_prefix = os.path.splitext(filename)[0]
        charts = generate_visualizations(df, analysis, output_prefix)
        story = narrate_story(api, analysis, charts, filename)
        readme_file = generate_unique_filename(f"README_{output_prefix}.md")
        save_readme(story, charts, readme_file)
        print(f"Analysis completed for {filename}. Check {readme_file} and charts.")

if __name__ == "__main__":
    main()