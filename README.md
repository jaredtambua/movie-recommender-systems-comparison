# Movie Recommender Systems: Hybrid Filtering vs Neural Collaborative Filtering

### Overview

This project implements and evaluates two movie recommendation systems using the MovieLens 1M dataset.

The objective is to compare a traditional recommendation approach against a modern neural-network-based recommender while providing an interactive recommendation experience through a command-line interface.

The project includes:

* Hybrid Recommender System (SVD + Content-Based Filtering)
* Neural Collaborative Filtering (NCF)
* Shared evaluation pipeline
* Pretrained model artefacts for immediate execution
* Interactive CLI for generating personalised movie recommendations

---

### Recommendation Systems

#### 1. Hybrid Recommender (SVD + Content-Based Filtering)

The hybrid system combines:

* Collaborative Filtering using Matrix Factorisation (SVD) with stochastic gradient descent.
* Content-Based Filtering using movie genre information

The collaborative component captures user-item interactions, while the content-based component helps address sparsity and cold-item situations.

#### 2. Neural Collaborative Filtering (NCF)

The neural recommender uses:

* User embeddings
* Movie embeddings
* User demographic features
* Movie genre features

A neural network learns latent user preferences and predicts ratings through non-linear interactions between users and movies.

---

### Dataset

This project uses the MovieLens 1M dataset.

MovieLens is a widely used benchmark dataset for recommender systems research and contains:

* 1 million movie ratings
* Approximately 6,000 users
* Approximately 4,000 movies

The dataset is not included in this repository.

To retrain the models from scratch, place the following files inside the `data/` directory:

```text
data/
├── movies.dat
├── ratings.dat
└── users.dat
```

Dataset source:

https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset

---

### Included Model Artefacts

Pretrained model artefacts are included in the repository.

This means the recommendation systems can be executed immediately without:

* Downloading MovieLens
* Rebuilding preprocessing pipelines
* Retraining models

The repository loads saved artefacts from the `models/` directory during startup.

To retrain from scratch, remove the contents of `models/` and provide the MovieLens dataset in the `data/` directory.

---

### Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Launch the application:

```bash
python main.py
```

The application provides a menu allowing you to choose between:

1. Hybrid Recommender (SVD + Content-Based Filtering)
2. Neural Collaborative Filtering (NCF)

You can:

* Select a specific user ID
* Generate recommendations for a random user
* Browse recommendation results through the CLI

---

### Evaluation

Run the evaluation pipeline:

```bash
python -m src.evaluation.run_eval
```

The evaluation framework compares the recommender systems using metrics including:

* RMSE
* Serendipity

---

### Future Improvements

Potential extensions include:

* Web-based interface
* Real-time recommendation serving
* Additional recommendation metrics
* Deployment using Docker and cloud infrastructure

---

### Technologies Used

* Python
* NumPy
* Pandas
* PyTorch
* MovieLens 1M Dataset
* Matrix Factorisation (SVD)
* Content-Based Filtering
* Neural Collaborative Filtering

--- 

### Demo Screenshots

#### Main Menu

![Main Menu](docs/screenshots/main_menu.png)

#### Hybrid Recommendations

![Hybrid Recommendations](docs/screenshots/hybrid_recommendations.png)

#### NCF Recommendations

![NCF Recommendations](docs/screenshots/ncf_recommendations.png)
