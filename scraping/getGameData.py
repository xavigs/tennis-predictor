from cassandra.cluster import Cluster
from bs4 import BeautifulSoup
import requests
import argparse

# Variables
parser = argparse.ArgumentParser()
parser.add_argument("season")
args = parser.parse_args()
season = args.season

# Functions

# Get Tournaments from DB
cluster = Cluster(["127.0.0.1"])
session = cluster.connect("beast")
query = "SELECT * FROM tournament WHERE tournament_season = " + season
tournaments = session.execute(query)

# Close connections
session.shutdown()
cluster.shutdown()

# Web scraping - Games
for tournament in tournaments:
    url = "http://www.tennisexplorer.com/" + tournament.tournament_keyword + "/" + season + "/atp-men/";
    print(url)
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data, "html.parser")

    for game in list(soup.select("table.result"))[0].select("tr[id^=r]"):
        print("Partit")
