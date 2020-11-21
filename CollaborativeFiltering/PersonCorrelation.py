import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from DataBase.DB import DB


class PersonCorrelation:
    """Class responsible for Person Correlation"""
    def __init__(self, num_of_neighbors, main_user_id, algorithm_nbr=0):
        """Constructor
        Depending on the value of algorithm_nbr parameter, an algorithm is selected and the necessary data is created
        algorithm_nbr == 0 -> Item Item Collaborative Filtering
        algorithm_nbr != 0 -> User User Collaborative Filtering
        :param num_of_neighbors: Number of neighbors
        :param main_user_id: Target User Id
        :param algorithm_nbr: Algorithm selection
        """
        self.main_user_id = main_user_id
        self.algorithm_nbr = algorithm_nbr
        if algorithm_nbr != 0:
            self.all_data = DB.get_all_data().T
            self.norm_rates = self.get_norm_rates()
            self.main_user_norm_rates = self.get_main_user_norm_rates()
            self.rest_of_users_norm_rates = self.get_rest_of_users_norm_rates()
            self.neighbors = self.create_user_user_neighbors(num_of_neighbors)
        else:
            self.all_data = DB.get_all_data()
            self.norm_rates = self.get_norm_rates()

            self.norm_usr_watched_movies = self.get_norm_usr_watched_movies()
            self.norm_usr_unwatched_movies = self.get_nom_user_unwatched_movies()
            self.neighbors = self.create_item_item_neighbors(num_of_neighbors)

    def get_norm_rates(self):
        """Normalizes all ratings
        :return pandas dataframe with normalized ratings
        """
        return self.all_data.sub(self.all_data.mean(axis=1), axis=0).fillna(0)

    def get_norm_usr_watched_movies(self):
        """normalizes the ratings of videos watched by the user
        :return pandas dataframe with normalized ratings
        """
        watch_movies = self.all_data[self.main_user_id].dropna().index
        return self.norm_rates.loc[watch_movies, :]

    def get_nom_user_unwatched_movies(self):
        """normalizes the ratings of videos not watched by the user
        :return pandas dataframe with normalized ratings
        """
        unwatched_movies = self.all_data[self.all_data[self.main_user_id].isnull()].index.values
        return self.norm_rates.loc[unwatched_movies, :]

    def get_main_user_norm_rates(self):
        """Normalizes target user ratings
        :return pandas series with normalized ratings
        """
        return self.norm_rates.loc[self.main_user_id].values.reshape(1, -1)

    def get_rest_of_users_norm_rates(self):
        """Normalizes all rating without target user ratings
        :return pandas dataframe with normalized ratings
        """
        return self.norm_rates.drop([self.main_user_id])

    def get_cosine_similarities(self, start_index=None, end_index=None):
        """Depending on the algorithm chosen, it calculates the cosine similarity between oceans
        param start_index and end index are not required if User User algorithm selected
        :param start_index: start of the data portion
        :param end_index: end of the data portion
        :return: numpy array with similarities calculated
        """
        if self.algorithm_nbr != 0:
            return cosine_similarity(self.main_user_norm_rates, self.rest_of_users_norm_rates).T
        return cosine_similarity(self.norm_usr_watched_movies, self.norm_usr_unwatched_movies[start_index:end_index]).T

    def index_to_id(self, index_, movies_table):
        """Returns user / movie ids from given table
        :param index_: item index
        :param movies_table: table from we can get id
        :return: user / movie id
        """
        return movies_table.index[index_]

    def get_best_neighbors_from_sim(self, similarities, num_of_neighbors, rates_table):
        """Returns most similar neighbors
        :param similarities: similarities row
        :param num_of_neighbors: Number of neighbors
        :param rates_table: table from we can get id
        :return: dictionary where key is neighbor id and value is neighbor similarity
        """
        best_neighbors_ids = similarities.argsort()[-num_of_neighbors:][::-1]
        return {self.index_to_id(id_, rates_table): np.float64(similarities[id_]) for id_ in
                best_neighbors_ids}

    def create_user_user_neighbors(self, num_of_neighbors):
        """Returns user best neighbors
        :param num_of_neighbors: Number of user neighbors
        :return: dictionary where key is neighbor id and value is neighbor similarity
        """
        cosine_similarities = self.get_cosine_similarities().T
        return self.get_best_neighbors_from_sim(cosine_similarities[0], num_of_neighbors, self.rest_of_users_norm_rates)

    def create_item_item_neighbors(self, num_of_neighbors, data_portion=4000):
        """Returns item best neighbors
        :param num_of_neighbors: Number of item neighbors
        :param data_portion: Number of data portions
        :return: dictionary where key is neighbor id and value is neighbor similarity
        """
        begin = 0
        end = data_portion
        index = 0
        movies_best_neighbors = dict()
        index_range = range(0, len(self.norm_usr_unwatched_movies.index) + 1)
        while True:
            try:
                if begin >= len(index_range):
                    break
                try:
                    cosine_similarities = self.get_cosine_similarities(index_range[begin:end][0],
                                                                       index_range[begin:end][-1])
                except ValueError:
                    break
                for movie_sim in cosine_similarities:
                    unwatched_movie = self.index_to_id(index, self.norm_usr_unwatched_movies)
                    movies_best_neighbors[unwatched_movie] = self.get_best_neighbors_from_sim(
                        movie_sim, num_of_neighbors, self.norm_usr_watched_movies)
                    index += 1
                begin = end - 1
                end += data_portion
            except MemoryError:
                end -= data_portion
                data_portion //= 2
                end += data_portion
                print(f"Memory Error, reducing data portion to {data_portion}")
        return movies_best_neighbors

    def get_neighbors(self, id_):
        """Returns item neighbors
        :param id_: target item id
        """
        neighbors = self.neighbors[id_]
        neighbors.pop(id_, None)
        return neighbors
