# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT
committed to git due to size. Follow the download instructions below.

## Dataset 1: NarrativeQA (HELM subset)

### Overview
- **Source**: https://huggingface.co/datasets/lighteval/narrative_qa_helm
- **Size**: train 1102, validation 115, test 355
- **Format**: HuggingFace Dataset
- **Task**: Narrative question answering over long passages
- **Splits**: train/validation/test
- **License**: Not specified in the dataset card

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

dataset = load_dataset("lighteval/narrative_qa_helm")
dataset.save_to_disk("datasets/narrative_qa_helm")
```

### Loading the Dataset
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/narrative_qa_helm")
```

### Sample Data
See `datasets/narrative_qa_helm/samples/train_samples.json`.

### Notes
- This is a compact subset from HELM and is much smaller than the full NarrativeQA.

## Dataset 2: BookSum

### Overview
- **Source**: https://huggingface.co/datasets/kmfoda/booksum
- **Size**: train 9600, validation 1484, test 1431
- **Format**: HuggingFace Dataset
- **Task**: Long-form book/chapter summarization and narrative understanding
- **Splits**: train/validation/test
- **License**: Not specified in the dataset card

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

dataset = load_dataset("kmfoda/booksum")
dataset.save_to_disk("datasets/booksum")
```

### Loading the Dataset
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/booksum")
```

### Sample Data
See `datasets/booksum/samples/train_samples.json`.

### Notes
- Contains long passages that can stress entity tracking over extended context.
