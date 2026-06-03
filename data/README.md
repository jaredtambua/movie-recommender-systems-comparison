## Dataset

This project uses the MovieLens 1M dataset.

The dataset is not included in this repository due to file size considerations.

To regenerate the pretrained model artefacts from scratch, download the MovieLens 1M dataset and place the following files in this directory:

```text
data/
├── movies.dat
├── ratings.dat
└── users.dat
```

Dataset source:

https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset

### Notes

The repository includes pretrained model artefacts, so the dataset is not required for normal use.

You only need the dataset if you wish to:

* Retrain the Hybrid recommender
* Retrain the Neural Collaborative Filtering model
* Regenerate preprocessing artefacts
* Reproduce the training pipeline from scratch
