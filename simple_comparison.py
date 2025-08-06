#!/usr/bin/env python3
"""Simple model comparison script for sentiment analysis models."""

import yaml
from pathlib import Path

def load_results():
    """Load all evaluation results."""
    results_dir = Path("results")
    results = {}
    
    # Define model files and their display names
    model_files = {
        "cnn_evaluation_results.yaml": "Optimized CNN",
        "hybrid_evaluation_results.yaml": "Hybrid CNN+LSTM", 
        "transformer_evaluation_results.yaml": "Transformer",
        "transformer_improved_evaluation_results.yaml": "Improved Transformer",
        "transformer_pretrained_evaluation_results.yaml": "Transformer + Pre-trained",
        "transformer_augmented_evaluation_results.yaml": "Transformer + Augmented"
    }
    
    print("🔍 Loading model evaluation results...")
    
    for filename, display_name in model_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)
                    results[display_name] = data
                    print(f"✅ Loaded results for {display_name}")
            except Exception as e:
                print(f"⚠️  Error loading {filename}: {e}")
        else:
            print(f"⚠️  Missing results for {display_name}: {filename}")
    
    return results

def print_comparison(results):
    """Print comparison table."""
    print("\n🏆 SENTIMENT ANALYSIS MODEL COMPARISON")
    print("=" * 80)
    
    # Print header
    print(f"{'Model':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Parameters':<12}")
    print("-" * 80)
    
    # Sort by accuracy
    sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for model_name, data in sorted_models:
        accuracy = data['accuracy']
        precision = data['precision']
        recall = data['recall']
        f1 = data['f1']
        params = data['model_parameters']
        
        print(f"{model_name:<30} {accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {params:,}")
    
    print("-" * 80)
    
    # Find best model
    best_model = sorted_models[0]
    print(f"\n🥇 BEST MODEL: {best_model[0]}")
    print(f"   Accuracy: {best_model[1]['accuracy']:.3f}")
    print(f"   F1-Score: {best_model[1]['f1']:.3f}")
    print(f"   Parameters: {best_model[1]['model_parameters']:,}")
    
    # Top 3 models
    print(f"\n🏅 TOP 3 MODELS BY ACCURACY:")
    for i, (model_name, data) in enumerate(sorted_models[:3], 1):
        print(f"   {i}. {model_name}: {data['accuracy']:.3f} accuracy")
    
    # Efficiency analysis
    print(f"\n⚡ EFFICIENCY ANALYSIS (Accuracy per Million Parameters):")
    efficiency_data = []
    for model_name, data in sorted_models:
        efficiency = data['accuracy'] / (data['model_parameters'] / 1_000_000)
        efficiency_data.append((model_name, efficiency, data['accuracy'], data['model_parameters']))
    
    efficiency_data.sort(key=lambda x: x[1], reverse=True)
    for i, (model_name, efficiency, accuracy, params) in enumerate(efficiency_data, 1):
        print(f"   {i}. {model_name}: {efficiency:.2f} (Acc: {accuracy:.3f}, Params: {params:,})")
    
    # Architecture analysis
    print(f"\n🏗️  ARCHITECTURE ANALYSIS:")
    
    cnn_models = [m for m in sorted_models if 'CNN' in m[0]]
    lstm_models = [m for m in sorted_models if 'LSTM' in m[0] and 'Hybrid' not in m[0]]
    transformer_models = [m for m in sorted_models if 'Transformer' in m[0]]
    hybrid_models = [m for m in sorted_models if 'Hybrid' in m[0]]
    
    if cnn_models:
        print("📈 CNN Models:")
        for model_name, data in cnn_models:
            print(f"   • {model_name}: {data['accuracy']:.3f} accuracy")
    
    if lstm_models:
        print("🔄 LSTM Models:")
        for model_name, data in lstm_models:
            print(f"   • {model_name}: {data['accuracy']:.3f} accuracy")
    
    if transformer_models:
        print("⚡ Transformer Models:")
        for model_name, data in transformer_models:
            print(f"   • {model_name}: {data['accuracy']:.3f} accuracy")
    
    if hybrid_models:
        print("🔗 Hybrid Models:")
        for model_name, data in hybrid_models:
            print(f"   • {model_name}: {data['accuracy']:.3f} accuracy")

def save_comparison(results):
    """Save comparison to markdown file."""
    output = []
    output.append("# Sentiment Analysis Model Comparison\n")
    
    # Sort by accuracy
    sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    # Create table
    output.append("| Model | Accuracy | Precision | Recall | F1-Score | Parameters |")
    output.append("|-------|----------|-----------|--------|----------|------------|")
    
    for model_name, data in sorted_models:
        output.append(f"| {model_name} | {data['accuracy']:.3f} | {data['precision']:.3f} | {data['recall']:.3f} | {data['f1']:.3f} | {data['model_parameters']:,} |")
    
    output.append(f"\n## Best Model: {sorted_models[0][0]}")
    output.append(f"- Accuracy: {sorted_models[0][1]['accuracy']:.3f}")
    output.append(f"- F1-Score: {sorted_models[0][1]['f1']:.3f}")
    output.append(f"- Parameters: {sorted_models[0][1]['model_parameters']:,}")
    
    # Save to file
    with open("MODEL_COMPARISON.md", "w") as f:
        f.write("\n".join(output))
    
    print("📄 Comparison saved to MODEL_COMPARISON.md")

def main():
    """Main function."""
    results = load_results()
    
    if not results:
        print("❌ No evaluation results found!")
        return
    
    print(f"✅ Loaded results for {len(results)} models")
    
    print_comparison(results)
    save_comparison(results)

if __name__ == "__main__":
    main() 