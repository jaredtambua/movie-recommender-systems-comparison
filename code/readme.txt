
------------------------------------------

Movie Recommender Systems (Hybrid & NCF)

This project implements two movie recommender systems using the MovieLens 1M dataset:
- RS1 (Hybrid): Collaborative Filtering (SVD) + Content-Based Filtering (CBF)
- RS2 (NCF): Neural Collaborative Filtering using user metadata + item genres

Both systems run via a command-line interface and load pretrained model artefacts from code/cache/, so retraining is not required during normal execution.

------------------------------------------

Main files are located in the code/ directory:

cli.py
Shared command-line interface utilities

main.py
Entry point for RS1 (Hybrid: SVD + CBF)

main2.py
Entry point for RS2 (NCF)

hybrid_engine.py
Implementation of RS1

ncf_engine.py
Implementation of RS2

run_eval.py
Runs evaluation (RMSE + serendipity) on the shared cached test split

cache/
Cached artefacts (pretrained weights + shared dataset split). These are loaded at runtime.

------------------------------------------

Dataset Location

The MovieLens 1M dataset (only needed for one-time preprocessing/training) is expected at:

../dataset relative to code/, containing:

ratings.dat

movies.dat

users.dat

The dataset can be found here: 
https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset 

------------------------------------------

Running without the dataset:

If code/cache/ contains the pretrained artefacts (including the shared split file), then ../dataset is NOT required for:

python main.py
python main2.py
python run_eval.py

------------------------------------------

Running with the dataset:

The dataset is only required if you want to regenerate the cached artefacts (e.g. if cache/ is missing or you intentionally delete it to rebuild).



------------------------------------------

Installation: 

pip install -r requirements.txt

------------------------------------------

How to Run:
RS1 — Hybrid (SVD + CBF) 
- python main.py

RS2 — Neural Collaborative Filtering (NCF) 
- python main2.py

Evaluation (RMSE + serendipity)
- python run_eval.py

------------------------------------------

Valid User IDs for Testing

User IDs must exist in the sampled dataset used to train the pretrained models. The following user IDs are known to be valid (not exhaustive):

[5, 8, 9, 10, 11, 13, 15, 17, 18, 19, 22, 23, 24]

Alternatively, use the “random user” option in the CLI.
