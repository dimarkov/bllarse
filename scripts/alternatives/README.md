# Alternative Benchmarks

Alternative benchmarks for applying Bayesian loss functions to Vision Transformers, NLP text classification, language modeling, and semantic segmentation.

## Files

| File | Description |
|------|-------------|
| `calibration.py` | Unified ECE metrics for all task types |
| `vit_classification.py` | ViT fine-tuning with equimo |
| `text_classification.py` | BERT/DistilBERT with Flax |
| `language_modeling.py` | GPT-2 next-token prediction |
| `segmentation.py` | Per-pixel Bayesian classification |

## Usage Examples

### ViT Classification (CIFAR-10/100)
```bash
python scripts/alternatives/vit_classification.py \
    --model vit_small_patch14_dinov2 \
    --dataset cifar10 \
    --loss-fn IBProbit \
    --epochs 10 \
    --batch-size 64
```

### NLP Text Classification (SST-2)
```bash
# Install NLP dependencies first
uv sync --group alternatives

python scripts/alternatives/text_classification.py \
    --model distilbert-base-uncased \
    --dataset sst2 \
    --loss-fn IBProbit \
    --epochs 5
```

### Language Modeling (GPT-2)
```bash
python scripts/alternatives/language_modeling.py \
    --model gpt2 \
    --dataset wikitext \
    --loss-fn IBProbit \
    --epochs 3 \
    --max-samples 5000
```

## Calibration Metrics

| Task Type | Metrics |
|-----------|---------|
| Classification | Accuracy, ECE, NLL |
| Segmentation | mIoU, Pixel Accuracy, ECE |
| Language Modeling | Perplexity, Token ECE, NLL |

## Dependencies

Install NLP dependencies:
```bash
uv sync --group alternatives
```

This installs:
- `transformers[flax]>=4.35.0`
- `sentencepiece>=0.2.0`
