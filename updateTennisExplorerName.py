from cassandra.cluster import Cluster
from bs4 import BeautifulSoup
from pprint import pprint
import sys
sys.path.append(r'C:\Users\d_mas\Developer\The Beast\lib')
import utils

# Variables
players_db = dict()
countries = dict()

# Get players from DB
cluster = Cluster(["127.0.0.1"])
session = cluster.connect("beast")

query = "SELECT player_keyword, player_atpwt_id, player_name, player_country, player_rankdate, player_te_name, player_te_url FROM player_by_keyword"
players = session.execute(query)

for player in players:
    player_db = dict()
    player_db['name'] = player.player_name
    player_db['country'] = player.player_country
    player_db['rankdate'] = str(player.player_rankdate)
    player_db['keyword'] = player.player_keyword

    if player.player_te_name == "BLANK" and player.player_atpwt_id not in players_db:
        players_db[player.player_atpwt_id] = player_db

        if player.player_country in countries:
            countries[player.player_country] += 1
        else:
            countries[player.player_country] = 1

for country, count in countries.items():
    print(country, count)
