# Medical Image Severity Classification

This repository contains a medical image classification project for brain MRI scans.
It is designed to classify medical images into one of four categories:
- `glioma`
- `meningioma`
- `pituitary`
- `notumor`

## Project Structure

- `train.py` — train the classification model
- `predict.py` — run inference on new images
- `evaluate.py` — compute model metrics and generate reports
- `preprocessing.py` — data preparation and image preprocessing
- `transfer_model.py` — transfer learning model implementation
- `hybrid_model.py` — hybrid model implementation
- `fuzzy_system.py` — optional fuzzy logic component
- `main.py` — pipeline orchestrator
- `utils.py` — utility functions
- `gui.py` — graphical interface script
- `dataset/` — image training and testing data directories
- `saved_model/` — local saved weights and model artifacts (excluded from GitHub push)
- `results/` — output reports and prediction artifacts

## Dataset Layout

The dataset is organized as:

```
dataset/
  Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Train the model:
```bash
python train.py
```

Predict with trained model:
```bash
python predict.py
```

Evaluate model performance:
```bash
python evaluate.py
```

Run the GUI (if available):
```bash
python gui.py
```

## Notes

- Large model files and dataset directories are not included in the GitHub commit by design.
- Keep any local large files such as `saved_model/` and `dataset/` outside of version control or use Git LFS for them.
- The repository focuses on the code pipeline for training, prediction, and evaluation.

## License

Add a license file if you want to make this repository open source.
