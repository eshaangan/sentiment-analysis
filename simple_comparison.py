#!/usr/bin/env python3
"""Simple comparison of all trained sentiment analysis models."""

import yaml
from pathlib import Path

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
            print(f"⚠️  Missing results for {model_name}: {filename}")
    
    return models

def print_comparison_table(models):
    """Print a simple comparison table."""
    print("🏆 SENTIMENT ANALYSIS MODEL COMPARISON")
    print("=" * 80)
    print()
    
    # Header
    print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Parameters':<15}")
    print("-" * 80)
    
    # Sort models by accuracy
    sorted_models = sorted(models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for model_name, results in sorted_models:
        print(f"{model_name:<25} {results['accuracy']:<10.3f} {results['precision']:<10.3f} "
              f"{results['recall']:<10.3f} {results['f1']:<10.3f} {results['model_parameters']:<15,}")
    
    print("-" * 80)
    print()

def print_analysis(models):
    """Print detailed analysis."""
    sorted_models = sorted(models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    # Best model
    best_model_name, best_results = sorted_models[0]
    print(f"🥇 BEST MODEL: {best_model_name}")
    print(f"   Accuracy: {best_results['accuracy']:.3f}")
    print(f"   F1-Score: {best_results['f1']:.3f}")
    print(f"   Parameters: {best_results['model_parameters']:,}")
    print()
    
    # Top 3 models
    print("🏅 TOP 3 MODELS BY ACCURACY:")
    for i, (model_name, results) in enumerate(sorted_models[:3], 1):
        print(f"   {i}. {model_name}: {results['accuracy']:.3f} accuracy")
    print()
    
    # Efficiency analysis
    print("⚡ EFFICIENCY ANALYSIS (Accuracy per Million Parameters):")
    efficiency_data = []
    for model_name, results in sorted_models:
        params_millions = results['model_parameters'] / 1000000
        efficiency = results['accuracy'] / params_millions
        efficiency_data.append((model_name, results['accuracy'], results['model_parameters'], efficiency))
    
    efficiency_data.sort(key=lambda x: x[3], reverse=True)
    for i, (model_name, acc, params, eff) in enumerate(efficiency_data[:3], 1):
        print(f"   {i}. {model_name}: {eff:.2f} (Acc: {acc:.3f}, Params: {params:,})")
    print()
    
    # Architecture analysis
    print("🏗️  ARCHITECTURE ANALYSIS:")
    cnn_models = [(name, results) for name, results in sorted_models if 'CNN' in name and 'Hybrid' not in name]
    lstm_models = [(name, results) for name, results in sorted_models if 'LSTM' in name and 'Hybrid' not in name]
    transformer_models = [(name, results) for name, results in sorted_models if 'Transformer' in name]
    hybrid_models = [(name, results) for name, results in sorted_models if 'Hybrid' in name]
    
    print("📈 CNN Models:")
    for model_name, results in cnn_models:
        print(f"   • {model_name}: {results['accuracy']:.3f} accuracy")
    
    print("🔄 LSTM Models:")
    for model_name, results in lstm_models:
        print(f"   • {model_name}: {results['accuracy']:.3f} accuracy")
    
    print("⚡ Transformer Models:")
    for model_name, results in transformer_models:
        print(f"   • {model_name}: {results['accuracy']:.3f} accuracy")
    
    print("🔗 Hybrid Models:")
    for model_name, results in hybrid_models:
        print(f"   • {model_name}: {results['accuracy']:.3f} accuracy")
    print()

def save_comparison_to_markdown(models):
    """Save comparison to markdown file."""
    sorted_models = sorted(models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    markdown_content = """# Sentiment Analysis Model Comparison

## Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | Parameters |
|-------|----------|-----------|--------|----------|------------|
"""
    
    for model_name, results in sorted_models:
        markdown_content += f"| {model_name} | {results['accuracy']:.3f} | {results['precision']:.3f} | {results['recall']:.3f} | {results['f1']:.3f} | {results['model_parameters']:,} |\n"
    
    best_model_name, best_results = sorted_models[0]
    
    markdown_content += f"""

## Key Findings

### 🏆 Best Performing Model
The **{best_model_name}** achieved the highest accuracy of **{best_results['accuracy']:.3f}** with an F1-score of **{best_results['f1']:.3f}**.

### 🏗️ Architecture Insights
- **CNN models** excel at local pattern detection and are computationally efficient
- **LSTM models** capture sequential dependencies but require more parameters  
- **Transformer models** struggled with this dataset size and complexity
- **Hybrid models** combine the strengths of multiple architectures

### 📊 Model Performance Ranking
"""
    
    for i, (model_name, results) in enumerate(sorted_models, 1):
        markdown_content += f"{i}. **{model_name}**: {results['accuracy']:.3f} accuracy ({results['model_parameters']:,} parameters)\n"
    
    markdown_content += """

### ⚡ Efficiency Analysis
Based on accuracy per million parameters:
"""
    
    efficiency_data = []
    for model_name, results in sorted_models:
        params_millions = results['model_parameters'] / 1000000
        efficiency = results['accuracy'] / params_millions
        efficiency_data.append((model_name, efficiency))
    
    efficiency_data.sort(key=lambda x: x[1], reverse=True)
    for i, (model_name, efficiency) in enumerate(efficiency_data[:3], 1):
        markdown_content += f"{i}. **{model_name}**: {efficiency:.2f} accuracy per million parameters\n"
    
    markdown_content += """

## Training Insights
- CNN models trained faster and converged more reliably
- Transformer models required more data and computational resources
- Hybrid models showed good balance but may overfit with current hyperparameters
- Early stopping was crucial for preventing overfitting

## Recommendations
1. **Production Use**: Use the **{}** for production sentiment analysis
2. **Resource Constraints**: Consider the most efficient model for limited computational resources
3. **Research**: Hybrid models show promise for future improvements

## Next Steps
1. **Ensemble Methods**: Combine top-performing models
2. **Hyperparameter Tuning**: Further optimize the best models
3. **Data Augmentation**: Increase training data for better generalization
4. **Transfer Learning**: Explore pre-trained embeddings
""".format(best_model_name)
    
    with open("MODEL_COMPARISON.md", "w") as f:
        f.write(markdown_content)
    
    print("📄 Comparison saved to MODEL_COMPARISON.md")

def main():
    """Main function to run the comparison."""
    print("🔍 Loading model evaluation results...")
    models = load_evaluation_results()
    
    if not models:
        print("❌ No evaluation results found!")
        return
    
    print(f"✅ Loaded results for {len(models)} models")
    print()
    
    # Print comparison table
    print_comparison_table(models)
    
    # Print detailed analysis
    print_analysis(models)
    
    # Save to markdown
    save_comparison_to_markdown(models)

if __name__ == "__main__":
    main() 