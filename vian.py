import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import openai
import uvicorn

# Set up the AI Proxy token
openai.api_key = os.environ.get("AIPROXY_TOKEN")

# Function to load the CSV data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to get summary statistics and check for missing data
def get_summary_statistics(data):
    summary = {
        "missing_values": data.isnull().sum(),
        "summary_statistics": data.describe(),
        "correlation_matrix": data.corr(),
    }
    return summary

# Function to run basic EDA and clustering analysis
def basic_analysis(data):
    scaler = StandardScaler()
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data_scaled = scaler.fit_transform(data[numeric_columns])

    kmeans = KMeans(n_clusters=3)
    data['Cluster'] = kmeans.fit_predict(data_scaled)

    return data, kmeans

# Function to create charts
def create_visualizations(data, summary, output_dir):
    # Correlation heatmap
    corr_heatmap = sns.heatmap(summary["correlation_matrix"], annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()

    # Cluster distribution plot
    sns.pairplot(data, hue="Cluster", palette='Set1')
    plt.title('Cluster Distribution')
    plt.savefig(os.path.join(output_dir, "cluster_distribution.png"))
    plt.close()

# Function to generate the narrative using OpenAI's GPT
def generate_narrative(data, summary, output_dir):
    data_description = data.head().to_dict()
    summary_stats = summary["summary_statistics"].to_dict()
    
    prompt = f"""
    Given the dataset with the following columns and types: {list(data.columns)},
    please describe the dataset, provide an analysis, and suggest actionable insights.
    
    The summary statistics are as follows: {summary_stats}.
    The correlation matrix is as follows: {summary["correlation_matrix"]}.
    Please write a narrative that:
    - Describes the data.
    - Summarizes the analysis.
    - Gives insights on possible patterns and implications for decision making.
    - Include the following images in your output:
      1. Correlation heatmap: {os.path.join(output_dir, "correlation_heatmap.png")}
      2. Cluster distribution: {os.path.join(output_dir, "cluster_distribution.png")}
    """
    
    try:
        response = openai.Completion.create(
            model="gpt-4.0-mini",  # Specify the model version
            prompt=prompt,
            max_tokens=1500,
            temperature=0.7,
        )
        narrative = response.choices[0].text.strip()
        return narrative
    except openai.error.OpenAIError as e:
        return f"Error generating narrative: {str(e)}"

# Function to write results to README.md
def write_readme(narrative, output_dir):
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write("# Automated Analysis Report\n\n")
        f.write("### Narrative of the Data Analysis\n")
        f.write(narrative)

# Main function to run the entire analysis process
def main(file_path):
    # Load the dataset
    data = load_data(file_path)

    # Get summary statistics and perform basic analysis
    summary = get_summary_statistics(data)
    data_with_clusters, _ = basic_analysis(data)

    # Create directory for output files
    output_dir = os.path.splitext(file_path)[0]
    os.makedirs(output_dir, exist_ok=True)

    # Create visualizations
    create_visualizations(data_with_clusters, summary, output_dir)

    # Generate the narrative story
    narrative = generate_narrative(data_with_clusters, summary, output_dir)

    # Write the narrative and results to README.md
    write_readme(narrative, output_dir)

    print(f"Analysis completed. Check the output directory: {output_dir}")

# Run the script by calling the main function when executed via the command line
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
    else:
        dataset_file = sys.argv[1]
        main(dataset_file)
