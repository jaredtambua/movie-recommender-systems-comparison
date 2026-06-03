## Model Artefacts

This directory contains pretrained model artefacts and preprocessing outputs used by the recommendation systems.

Including these files allows the project to run immediately after cloning without requiring:

* Dataset download
* Data preprocessing
* Model retraining

### Contents

#### Hybrid Recommender

* `hybrid_artifacts.npz` — trained SVD parameters and content-based filtering artefacts
* `hybrid_metadata.pkl` — mappings, titles, and supporting metadata

#### Neural Collaborative Filtering

* `ncf_model.pt` — trained PyTorch model weights
* `ncf_arrays.npz` — precomputed arrays used by the NCF pipeline
* `ncf_meta.pkl` — metadata and mapping information

#### Shared Evaluation

* `preprocessed_shared.pkl` — shared train/validation/test split used for model evaluation

## Reproducing Results

To regenerate these artefacts from scratch:

1. Download the MovieLens 1M dataset.
2. Place the dataset files in the `data/` directory.
3. Delete the contents of this directory.
4. Run `main.py`.

If pretrained artefacts are not found, the project will automatically preprocess the dataset, train the recommender systems, and regenerate the required model files.