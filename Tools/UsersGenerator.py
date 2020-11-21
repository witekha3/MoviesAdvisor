import re
import time

import pandas as pd
import numpy as np
import requests
from DataBase.DB import DB
from requests_html import HTMLSession


class UsersGenerator:

    def __init__(self):
        self.session = HTMLSession()
        self.main_page = "https://www.imdb.com"
        self.path_to_ids_list = r"E:\projects\IMDbAdvisor\Tools\UsersIDsList.txt"
        self.user_pages_regex = r"class=\"flat-button lister-page-next next-page\" href=\"(.+?)\">.*?Next.*?</a>"
        self.user_star_regex = r"href=\"/title/tt(.*?)/.*?ipl-rating-star--other-user small.+" \
                               r"?ipl-rating-star__rating\">(.+?)</span>"

    def get_from_IMDb(self, url, regex):
        while True:
            try:
                r = self.session.get(url, timeout=15)
                response = str(r.text).replace("\n", "")
                if "Error 503" in response:
                    print("error 503. Suggests proxy change. Waiting 10s...")
                    time.sleep(10)
                    continue
                return re.findall(regex, response)
            except requests.exceptions.ConnectionError:
                print("Connection error. Changing proxy...")
            except requests.exceptions.ChunkedEncodingError:
                print("Chunked Encoding error, continuing...")
                return -1

    def get_user_movies_from_url(self, url):
        user_movies = dict()
        movies_ids_and_rates = self.get_from_IMDb(url, self.user_star_regex)
        if movies_ids_and_rates != -1:
            if len(movies_ids_and_rates) > 0:
                for movie_id_and_rate in movies_ids_and_rates:
                    if ">" not in movie_id_and_rate[0]:
                        user_movies[movie_id_and_rate[0]] = np.float32(movie_id_and_rate[1])
            return user_movies

    def get_all_user_movies(self, user_id):
        url = f'{self.main_page}/user/ur{user_id}/ratings'
        all_movies = dict()
        while True:
            all_movies.update(self.get_user_movies_from_url(url))
            response = self.get_from_IMDb(url, self.user_pages_regex)
            if response == -1:
                return []
            if response == "#" or len(response) == 0:
                break
            url = self.main_page + response[0]
        return all_movies

    def update_user(self, user_id, one_update=False):
        all_movies = self.get_all_user_movies(user_id)
        is_added = False
        if len(all_movies) > 0:
            user_pd = pd.DataFrame(all_movies, index=[user_id], columns=all_movies.keys()).T
            is_added = DB.add_user(user_pd)
        if one_update and is_added is True:
            DB.commit_database()

    def fill_database(self):
        with open(self.path_to_ids_list, "r") as f:
            lines = f.readlines()
        users_ids = {line.strip() for line in lines}
        for user_id in users_ids:
            if not DB.check_if_user_exist(user_id):
                self.update_user(user_id)
        DB.commit_database()
