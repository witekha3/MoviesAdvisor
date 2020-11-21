import math
import operator
import time

import numpy as np
from imdb import IMDb, IMDbDataAccessError

from CollaborativeFiltering.PersonCorrelation import PersonCorrelation


class UserUserCF(PersonCorrelation):
    """Class recommending the best product using the User User Collaborative Filtering algorithm"""
    def __init__(self, num_of_neighbors, main_user_id):
        """Constructor inheriting from the PersonCorrelation class"""
        super().__init__(num_of_neighbors, main_user_id, 1)

    def get_only_neighbors_movies(self):
        """Returns movies watched by neighbors with target user did not watched"""
        def get_usr_movies_ids(usr):
            return set(self.all_data.loc[usr].dropna().index.tolist())
        movies = set()
        user_movies = get_usr_movies_ids(self.main_user_id)
        for neighbor in self.neighbors:
            neighbor_movies = get_usr_movies_ids(neighbor)
            movies.update(neighbor_movies.difference(user_movies))
        return movies

    def calculate_rating_prediction(self, movie_id):
        """Returns rating prediction for given item
        :param movie_id: movie id
        :return: rating prediction for movie
        """
        score = 0
        similarity_sum = 0
        for neighbor, similarity in self.neighbors.items():
            movie_neighbor_rate = self.all_data.at[neighbor, movie_id]
            if math.isnan(movie_neighbor_rate):
                movie_neighbor_rate = 0
            score += similarity * movie_neighbor_rate
            similarity_sum += similarity
        return score / similarity_sum

    @staticmethod
    def _get_move_and_year(imdb, best_movie):
        while True:
            try:
                movie = imdb.get_movie(best_movie[0])
                return movie, str(movie['year'])
            except (KeyError, IMDbDataAccessError):
                print("Error, retrying...")
                time.sleep(1)
                continue


    def get_best_movies(self, num_of_movies):
        """Returns movies with highest values"""
        unwatched_movies_ratings_prediction = []
        imdb = IMDb()
        best_movies_message = ""
        for movie in self.get_only_neighbors_movies():
            score = self.calculate_rating_prediction(movie)
            unwatched_movies_ratings_prediction.append([movie, score])
        for i in range(1, num_of_movies+1):
            best_movie = max(unwatched_movies_ratings_prediction, key=operator.itemgetter(1))
            #print(best_movie)
            movie, year = UserUserCF._get_move_and_year(imdb, best_movie)
            best_movies_message += f"{i}: \t{movie} ({year}).\n\tPredicted rate: {best_movie[1]}\n"
            unwatched_movies_ratings_prediction.remove(best_movie)
        return best_movies_message
