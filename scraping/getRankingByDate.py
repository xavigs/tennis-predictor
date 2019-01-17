#!/usr/bin/env python
# coding: utf-8

# In[67]:


from cassandra.cluster import Cluster
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
#import argparse

# Variables
#parser = argparse.ArgumentParser()
#parser.add_argument("date")
#args = parser.parse_args()
#date = args.date
players_db = []
dates = []
current_date = "2013-09-30"
last_date = "2017-12-25"

# Functions
def addslashes(s):
    l = ["\\", '"', "'", "\0", ]
    for i in l:
        if i in s:
            s = s.replace(i, i+i)
    return s

def search_td(matrix, key1, value1, key2, value2):
    for index, dictionary in enumerate(matrix):
        if dictionary[key1] == value1 and dictionary[key2] == value2:
            return index
    
    return -1

# Dates
while current_date <= last_date:
    dates.append(current_date)
    params = current_date.split("-")
    current_datetime = datetime(int(params[0]), int(params[1]), int(params[2]))
    current_datetime += timedelta(days=7)
    current_date = current_datetime.strftime("%Y-%m-%d")

# Web scraping - Ranking
for date in dates:
    print("Scanning the ranking of " + date + "...");
    url = "https://www.atptour.com/en/rankings/singles?rankDate=" + date + "&rankRange=1-5000"
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data, "html.parser")

    for player in soup.select("table.mega-table tbody tr"):
        player_db = dict()
        player_db['date'] = date
        player_data = list(player.select("td"))
        player_db['ranking'] = player_data[0].text.strip().replace("T", "")
        player_db['country'] = list(player_data[2].select("img"))[0]['alt']
        player_db['name'] = addslashes(player_data[3].text.strip())
        link = list(player_data[3].select("a"))[0]['href']
        params = link.split("/")
        player_db['keyword'] = addslashes(params[3])
        player_db['atpwt_id'] = params[4].upper()
        players_db.append(player_db)

# Web scraping - Race
for date in dates:
    print("Scanning the race of " + date + "...");
    url = "http://www.atpworldtour.com/en/rankings/singles-race-to-london?rankDate=" + date + "&rankRange=1-5000"
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data, "html.parser")

    for player in soup.select("table.mega-table tbody tr"):
        player_data = list(player.select("td"))
        race = player_data[0].text.strip().replace("T", "")
        link = list(player_data[3].select("a"))[0]['href']
        params = link.split("/")
        atpwt_id = params[4].upper()
        key_player = search_td(players_db, "atpwt_id", atpwt_id, "date", date)
        players_db[key_player]['race'] = player_data[0].text.strip().replace("T", "")

# Insert
cluster = Cluster(["127.0.0.1"])
session = cluster.connect("beast")

for player_db in players_db:
    if "race" not in player_db:
        player_db['race'] = "0"
    insert = """INSERT INTO player_week (player_keyword, player_atpwt_id, player_name, player_country, player_ranking, player_race, player_rankdate, "player_C_career", "player_C_year", "player_C_season", "player_G_career", "player_G_year", "player_G_season", "player_H_career", "player_H_year", "player_H_season", "player_I_career", "player_I_year", "player_I_season", player_birth, player_rankmax, player_te_name, player_te_url) VALUES ('""" + player_db['keyword'] + "', '" + player_db['atpwt_id'] + "', '" + player_db['name'] + "', '" + player_db['country'] + "', " + player_db['ranking'] + ", " + player_db['race'] + ", '" + player_db['date'] + "', -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, '1900-01-01', -1, 'BLANK', 'BLANK')"
    print(insert)
    session.execute(insert)

# Close connections
session.shutdown()
cluster.shutdown()

