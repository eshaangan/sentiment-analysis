# Sentiment Analysis Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Parameters |
|-------|----------|-----------|--------|----------|------------|
| Transformer + Augmented | 0.850 | 0.851 | 0.850 | 0.850 | 2,106,626 |
| Optimized CNN | 0.838 | 0.840 | 0.838 | 0.838 | 1,141,218 |
| Hybrid CNN+LSTM | 0.811 | 0.812 | 0.811 | 0.811 | 2,169,667 |
| Transformer + Pre-trained | 0.782 | 0.796 | 0.782 | 0.779 | 1,438,210 |
| Transformer | 0.670 | 0.685 | 0.670 | 0.661 | 1,826,690 |
| Improved Transformer | 0.511 | 0.261 | 0.511 | 0.346 | 7,068,930 |

## Best Model: Transformer + Augmented
- Accuracy: 0.850
- F1-Score: 0.850
- Parameters: 2,106,626