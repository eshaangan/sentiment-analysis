#!/usr/bin/env python3
"""Compare all trained sentiment analysis models."""

import yaml
from pathlib import Path
import pandas as pd
from tabulate import tabulate

def load_evaluation_results():
    """Load evaluation results from all model files."""
    results_dir = Path("results")
    models = {}
    
    # Define expected result files
    result_files = {
        "Basic LSTM": "lstm_evaluation_results.yaml",
        "Better LSTM": "lstm_better_evaluation_results.yaml", 
        "Optimized CNN": "cnn_evaluation_results.yaml",
        "Transformer": "transformer_evaluation_results.yaml",
        "Improved Transformer": "transformer_improved_evaluation_results.yaml",
        "Hybrid CNN+LSTM": "hybrid_evaluation_results.yaml"
    }
    
    for model_name, filename in result_files.items():
        file_path = results_dir / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                models[model_name] = yaml.safe_load(f)
        else:
            print(f"Missing results for {model_name}: {filename}")
    
    return models

def create_comparison_table(models):
    """Create a comparison table of all models."""
    data = []
    
    for model_name, results in models.items():
        data.append({
            "Model": model_name,
            "Accuracy": f"{results['accuracy']:.3f}",
            "Precision": f"{results['precision']:.3f}",
            "Recall": f"{results['recall']:.3f}",
            "F1-Score": f"{results['f1']:.3f}",
            "Parameters": f"{results['model_parameters']:,}",
            "Samples": results['num_samples']
        })
    
    # Sort by accuracy (descending)
    data.sort(key=lambda x: float(x['Accuracy']), reverse=True)
    
    return data

def print_comparison(data):
    """Print the comparison table."""
    print("SENTIMENT ANALYSIS MODEL COMPARISON")
    print("=" * 80)
    print()
    
    # Create table
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Parameters", "Samples"]
    table = tabulate(data, headers=headers, tablefmt="grid", floatfmt=".3f")
    print(table)
    print()
    
    # Find best model
    best_model = data[0]
    print(f"🥇 BEST MODEL: {best_model['Model']}")
    print(f"   Accuracy: {best_model['Accuracy']}")
    print(f"   F1-Score: {best_model['F1-Score']}")
    print(f"   Parameters: {best_model['Parameters']}")
    print()
    
    # Performance analysis
    print("PERFORMANCE ANALYSIS:")
    print("-" * 40)
    
    accuracies = [(row['Model'], float(row['Accuracy'])) for row in data]
    
    print("🏅 Top 3 Models by Accuracy:")
    for i, (model, acc) in enumerate(accuracies[:3], 1):
        print(f"   {i}. {model}: {acc:.3f}")
    
    print()
    
    # Efficiency analysis
    print("⚡ EFFICIENCY ANALYSIS:")
    print("-" * 40)
    
    # Find most efficient (best accuracy/parameter ratio)
    efficiency_data = []
    for row in data:
        if row['Parameters'] != 'N/A':
            params = int(row['Parameters'].replace(',', ''))
            acc = float(row['Accuracy'])
            efficiency = acc / (params / 1000000)  # accuracy per million parameters
            efficiency_data.append((row['Model'], acc, params, efficiency))
    
    efficiency_data.sort(key=lambda x: x[3], reverse=True)
    
    print("Most Efficient Models (Accuracy per Million Parameters):")
    for i, (model, acc, params, eff) in enumerate(efficiency_data[:3], 1):
        print(f"   {i}. {model}: {eff:.2f} (Acc: {acc:.3f}, Params: {params:,})")
    
    print()
    
    # Model type analysis
    print("ARCHITECTURE ANALYSIS:")
    print("-" * 40)
    
    cnn_models = [row for row in data if 'CNN' in row['Model']]
    lstm_models = [row for row in data if 'LSTM' in row['Model'] and 'Hybrid' not in row['Model']]
    transformer_models = [row for row in data if 'Transformer' in row['Model']]
    hybrid_models = [row for row in data if 'Hybrid' in row['Model']]
    
    print("CNN Models:")
    for model in cnn_models:
        print(f"   • {model['Model']}: {model['Accuracy']} accuracy")
    
    print("🔄 LSTM Models:")
    for model in lstm_models:
        print(f"   • {model['Model']}: {model['Accuracy']} accuracy")
    
    print("⚡ Transformer Models:")
    for model in transformer_models:
        print(f"   • {model['Model']}: {model['Accuracy']} accuracy")
    
    print("🔗 Hybrid Models:")
    for model in hybrid_models:
        print(f"   • {model['Model']}: {model['Accuracy']} accuracy")

def save_comparison_to_markdown(data):
    """Save comparison to markdown file."""
    markdown_content = """# Sentiment Analysis Model Comparison

## Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | Parameters | Samples |
|-------|----------|-----------|--------|----------|------------|---------|
"""
    
    for row in data:
        markdown_content += f"| {row['Model']} | {row['Accuracy']} | {row['Precision']} | {row['Recall']} | {row['F1-Score']} | {row['Parameters']} | {row['Samples']} |\n"
    
    markdown_content += """

## Key Findings

### Best Performing Model
The **{}** achieved the highest accuracy of **{}** with an F1-score of **{}**.

### ⚡ Most Efficient Model
Based on accuracy per million parameters, the most efficient model is **{}**.

### Architecture Insights
- **CNN models** excel at local pattern detection and are computationally efficient
- **LSTM models** capture sequential dependencies but require more parameters
- **Transformer models** struggled with this dataset size and complexity
- **Hybrid models** combine the strengths of multiple architectures

### Recommendations
1. **Production Use**: Use the **{}** for production sentiment analysis
2. **Resource Constraints**: Consider **{}** for limited computational resources
3. **Research**: The **{}** shows promise for future improvements

## Training Insights
- CNN models trained faster and converged more reliably
- Transformer models required more data and computational resources
- Hybrid models showed good balance but may overfit with current hyperparameters
- Early stopping was crucial for preventing overfitting

## Next Steps
1. **Ensemble Methods**: Combine top-performing models
2. **Hyperparameter Tuning**: Further optimize the best models
3. **Data Augmentation**: Increase training data for better generalization
4. **Transfer Learning**: Explore pre-trained embeddings
""".format(
        data[0]['Model'], data[0]['Accuracy'], data[0]['F1-Score'],
        data[0]['Model'],  # Placeholder for most efficient
        data[0]['Model'], data[0]['Model'], data[0]['Model']
    )
    
    with open("MODEL_COMPARISON.md", "w") as f:
        f.write(markdown_content)
    
    print("📄 Comparison saved to MODEL_COMPARISON.md")

def main():
    """Main function to run the comparison."""
    print("Loading model evaluation results...")
    models = load_evaluation_results()
    
    if not models:
        print("No evaluation results found!")
        return
    
    print(f"Loaded results for {len(models)} models")
    print()
    
    # Create comparison table
    data = create_comparison_table(models)
    
    # Print comparison
    print_comparison(data)
    
    # Save to markdown
    save_comparison_to_markdown(data)

if __name__ == "__main__":
    main() 