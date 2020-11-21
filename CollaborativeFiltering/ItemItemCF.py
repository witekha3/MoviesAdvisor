import operator
import time

from imdb import IMDb, IMDbDataAccessError

from CollaborativeFiltering.PersonCorrelation import PersonCorrelation


class ItemItemCF(PersonCorrelation):
    """Class recommending the best product using the Item Item Collaborative Filtering algorithm"""
    def __init__(self, num_of_neighbors, main_user_id):
        """Constructor inheriting from the PersonCorrelation class"""
        super().__init__(num_of_neighbors, main_user_id)
        self.usr_unwatched_movies = self.all_data[self.all_data[main_user_id].isnull()].index.values

    def calculate_rating_prediction(self, main_id):
        """Returns rating prediction for given item
        :param main_id: itme id
        :return: rating prediction for item
        """
        neighbors = self.get_neighbors(main_id)
        score = 0
        similarity_sum = 0
        for neighbor_id, similarity in neighbors.items():
            if similarity == 0:
                return 0
            actual_rate = self.all_data.at[neighbor_id, self.main_user_id]
            score += actual_rate * similarity
            similarity_sum += similarity
        score /= similarity_sum
        return score

    def get_unwatched_movies_ratings_prediction(self):
        """Returns dictionary where key is item id and value is rating prediction"""
        return {main_id: self.calculate_rating_prediction(main_id) for main_id in self.usr_unwatched_movies}

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
        unwatched_movies_ratings_prediction = list(self.get_unwatched_movies_ratings_prediction().items())
        best_movies_message = ""
        imdb = IMDb()
        for i in range(1, num_of_movies+1):
            best_movie = max(unwatched_movies_ratings_prediction, key=operator.itemgetter(1))
            print(best_movie)
            movie, year = ItemItemCF._get_move_and_year(imdb, best_movie)
            best_movies_message += f"{i}: \t{movie} ({year}).\n\tPredicted rate: {best_movie[1]}\n"
            unwatched_movies_ratings_prediction.remove(best_movie)
        return best_movies_message
