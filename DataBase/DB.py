import pandas as pd
import numpy as np

class DB:
    _db = pd.read_pickle(r"DataBase/db.pkl").astype(np.float32)
    #_db = pd.read_pickle(r"Tests/inz_db.pkl")
    #_db = pd.read_pickle(r"E:\projects\IMDbAdvisor\Tests\new_test.pkl").astype(np.float32)
    #_db = pd.read_pickle(r"E:\projects\IMDbAdvisor\Tests\item_item_test_db.pkl")
    #_db = pd.read_pickle(r"E:\projects\IMDbAdvisor\Tests\user_user_test_db.pkl")

    @staticmethod
    def check_if_user_exist(user_id):
        return user_id in DB._db.columns

    @staticmethod
    def add_user(user_df):
        user_id = user_df.columns[0]
        if not DB.check_if_user_exist(user_id):
            DB._db = pd.concat([DB._db, user_df], axis=1)
            print(f"User {user_id} added!")
            return True
        if not DB._db[user_id].dropna().sort_index().equals(user_df[user_id].sort_index()):
            DB._db.drop(columns=user_id, inplace=True)
            DB._db = pd.concat([DB._db, user_df], axis=1)
            print(f"User {user_id} updated!")
            return True
        else:
            print(f"User {user_id} exist!")
            return False

    @staticmethod
    def commit_database():
        print("Saving changes in DB...")
        DB._db.to_pickle("DataBase/db.pkl")

    @staticmethod
    def get_all_data():
        return DB._db
