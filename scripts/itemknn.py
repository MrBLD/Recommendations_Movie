import numpy as np
import pandas as pd
import re
import os
import heapq
from surprise import accuracy
from surprise import Dataset
from surprise import Reader
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import NormalPredictor, KNNBasic
from recommender_metrics import RecommenderMetrics
from movie_lens_data import MovieLensData
from evaluator import Evaluator
path = "./ml-100k"
movie_lens_data = MovieLensData(
    users_path = os.path.join(path, "u.user"),
    ratings_path = os.path.join(path, "u.data"), 
    movies_path = os.path.join(path, "u.item"), 
    genre_path = os.path.join(path, "u.genre") 
    )

evaluation_data = movie_lens_data.read_ratings_data()
movie_data = movie_lens_data.read_movies_data()
popularity_rankings = movie_lens_data.get_popularity_ranks()
ratings = movie_lens_data.get_ratings()
evaluator = Evaluator(evaluation_data, popularity_rankings)
# adding random evaluator
algo_np = NormalPredictor()
evaluator.add_algorithm(algo_np, "Random")
# Add item-based collaborative filtering RecSys to evaluator


#Using the sim_options, we specify the type of similarity calculation and if the collaborative filtering is user based (in this case, No)
item_KNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': False})
evaluator.add_algorithm(item_KNN, "Item KNN")
ItemKnn
user_KNN = KNNBasic(sim_options = {'name': 'pearson', 'user_based': True})
evaluator.add_algorithm(user_KNN, "User KNN")
evaluator.evaluate(do_top_n=False)
# Time consuming, uncomment optionally
evaluator.evaluate(do_top_n=True)
# Evaluate topN recommendations
evaluator.sample_top_n_recs(movie_lens_data, test_subject=85, k=10)
evaluator.sample_top_n_recs(movie_lens_data, test_subject=85, k=5)
evaluator.sample_top_n_recs(movie_lens_data, test_subject=314, k=10)