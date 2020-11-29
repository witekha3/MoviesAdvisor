from tkinter import Tk, Label, Entry, Radiobutton, StringVar, Button, Text, IntVar, INSERT, END
from tkinter.scrolledtext import ScrolledText

from CollaborativeFiltering.ItemItemCF import ItemItemCF
from CollaborativeFiltering.UserUserCF import UserUserCF
from Tools import Config
from Tools.UsersGenerator import UsersGenerator


class AdvisorGui:

    def __init__(self):
        self.root = Tk()
        self.root.geometry("600x400")
        self.imdb_id_entry = None
        self.num_of_neighbors_entry = None
        self.num_of_common_movies_entry = None
        self.log_area = None
        self.algorithm_nbr = IntVar()
        self.num_of_neighbors = 4

    def start(self):
        self.init_id_placeholder()
        self.init_algorithms_selectors()
        self.init_num_of_neighbors_placeholder()
        self.init_num_of_movie_placeholder()
        self.init_info_area()
        self.init_start_button()
        self.root.mainloop()

    def init_id_placeholder(self):
        Label(self.root, text="Insert your IMDb ID: ").grid(row=0, column=0, sticky="W")
        default_id = IntVar()
        self.imdb_id_entry = Entry(textvariable=default_id)
        self.imdb_id_entry.grid(row=0, column=1, sticky="W")
        default_id.set(Config.main_user_id)

    def init_algorithms_selectors(self):
        Label(self.root, text="Select algorithm: ").grid(row=1, column=0, sticky="W")
        Radiobutton(self.root, text="User-User", variable=self.algorithm_nbr, value=1).grid(row=1, column=1, sticky="W")
        Radiobutton(self.root, text="Item-Item", variable=self.algorithm_nbr, value=0).grid(row=1, column=2, sticky="W")

    def init_num_of_neighbors_placeholder(self):
        Label(self.root, text="Set num of neighbors: ").grid(row=3, column=0, sticky="W")
        default_num = IntVar()
        self.num_of_neighbors_entry = Entry(textvariable=default_num)
        self.num_of_neighbors_entry.grid(row=3, column=1, sticky="W")
        default_num.set(Config.num_of_neighbors)

    def init_num_of_movie_placeholder(self):
        Label(self.root, text="Set num of movies you want: ").grid(row=4, column=0, sticky="W")
        default_num = IntVar()
        self.num_of_common_movies_entry = Entry(textvariable=default_num)
        self.num_of_common_movies_entry.grid(row=4, column=1, sticky="W")
        default_num.set(Config.num_of_movies)

    def init_info_area(self):
        self.log_area = ScrolledText(self.root, state='disabled', width=70, height=15)
        self.log_area.grid(column=0, row=5, sticky='w', columnspan=4, padx=12, pady=10)

    def init_start_button(self):
        Button(self.root, text="Find movies!", command=self.start_searching).grid(row=10, column=1)

    def start_searching(self):
        Config.num_of_neighbors = int(self.num_of_neighbors_entry.get())
        Config.num_of_movies = int(self.num_of_common_movies_entry.get())
        Config.algorithm_nbr = self.algorithm_nbr.get()
        Config.main_user_id = self.imdb_id_entry.get()
        UsersGenerator().update_user(one_update=True)
        if Config.algorithm_nbr == 1:
            best_movies = UserUserCF().get_best_movies()
        else:
            best_movies = ItemItemCF().get_best_movies()
        self.log_area.configure(state='normal')
        self.log_area.delete(1.0, END)
        self.log_area.insert(INSERT, best_movies)
        self.log_area.configure(state='disabled')
