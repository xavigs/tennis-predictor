from cassandra.cluster import Cluster
from bs4 import BeautifulSoup
import re
import requests
import pycountry
from pprint import pprint
import sys
sys.path.append(r'C:\Users\d_mas\Developer\The Beast\lib')
import utils
from colorama import init, Fore, Back, Style
from datetime import datetime

# Variables
init() # Init colorama
players_db = dict()
players_te = []
countries_pycountry = ["Bolivia, Plurinational State of", "Bosnia and Herzegovina", "Czechia", "Dominican Republic", "United Kingdom", "Macedonia, Republic of", "Moldova, Republic of", "Papua New Guinea", "South Africa", "Russian Federation", "Korea, Republic of", "Taiwan, Province of China", "Tunisia", "United States", "Venezuela, Bolivarian Republic of", "Viet Nam"]
countries_te = ["Bolivia", "Bosnia and Herzeg.", "Czech Republic", "Dominican Rep.", "Great Britain", "Macedonia", "Moldavsko", "Papua N. Guinea", "RSA", "Russia", "South Korea", "Taipei (CHN)", "Tunis", "USA", "Venezuela", "Vietnam"]
abbr_pycountry = ["BGR", "BRB", "CHE", "CHL", "DEU", "DNK", "GRC", "HRV", "LVA", "MCO", "NLD", "OMN", "PRI", "PRT", "PRY", "SLV", "SVN", "TWN", "URY", "VNM", "ZAF", "ZWE"]
abbr_atp = ["BUL", "BAR", "SUI", "CHI", "GER", "DEN", "GRE", "CRO", "LAT", "MON", "NED", "OMA", "PUR", "POR", "PAR", "ESA", "SLO", "TPE", "URU", "VIE", "RSA", "ZIM"]
page = 1

# Get players from DB
cluster = Cluster(["127.0.0.1"])
session = cluster.connect("beast")

#query = "SELECT player_atpwt_id, player_name, player_country FROM player_week WHERE player_rankdate = '2013-09-30'" # Test
query = "SELECT player_keyword, player_atpwt_id, player_name, player_country, player_rankdate FROM player_by_keyword" # Production
players = session.execute(query)

for player in players:
    player_db = dict()
    player_db['name'] = player.player_name
    player_db['country'] = player.player_country
    player_db['rankdate'] = str(player.player_rankdate)
    player_db['keyword'] = player.player_keyword

    if not player.player_atpwt_id in players_db and player.player_name != None:
        players_db[player.player_atpwt_id] = player_db

# Get players by country from Tennis Explorer
url = "https://www.tennisexplorer.com/list-players/"
r = requests.get(url)
data = r.text
soup = BeautifulSoup(data, "html.parser")
countries = soup.select("tbody#rank-country td a")

for index, country in enumerate(countries):
    # Test from specific country
    if country.text.strip() and index == 36:
        country_pycountry = pycountry.countries.get(name=country.text.strip())

        if country_pycountry is None:
            country_pycountry = pycountry.countries.get(name=utils.replaceMultiple2(country.text.strip(), countries_te, countries_pycountry))

        # List players from country (DB)
        country_players = []

        for atp_id, player in players_db.items():
            if player['country'] == utils.replaceMultiple2(country_pycountry.alpha_3, abbr_pycountry, abbr_atp):
                country_players.append(atp_id)
        pprint(country_players)

        print("Nº de jugadors: " + str(len(country_players)))

        # Web scraping - Country players list from Tennis Explorer
        end_pages = False
        while not end_pages:
            url = "https://www.tennisexplorer.com/list-players/" + country.get('href') + "&page=" + str(page) + "&order=rank"
            r = requests.get(url)
            data = r.text
            soup = BeautifulSoup(data, "html.parser")

            # Validate if there are players
            content = list(soup.select("form#playerSearch"))[0].parent.text.strip()

            if "No players" in content:
                end_pages = True
            else:
                print(Back.BLUE + "\n--- PÀGINA " + str(page) + " ---")
                print(Style.RESET_ALL)

                for player in soup.select("tbody.flags tr"):
                    if list(player.select("td"))[1].text.strip() == "":
                        end_pages = True

                    if not end_pages:
                        te_name = list(player.select("td"))[1].text.strip().split(", ")
                        atp_id = utils.searchKeyDictionaryByValue(players_db, "name", te_name[1] + " " + te_name[0], True)

                        if atp_id and te_name[1] + " " + te_name[0] != "Hao Zhang" and te_name[1] + " " + te_name[0] != "Sheng Hao Jin":
                            print("Jugador localitzat: " + te_name[1] + " " + te_name[0] + "!!! (" + players_db[atp_id]['rankdate'] + ") - " + list(player.select("a"))[0]['href'])
                            print(atp_id)

                            try:
                                country_players.remove(atp_id)
                                player_te = dict()
                                player_te['keyword'] = players_db[atp_id]['keyword'].replace("'", "''")
                                player_te['te_name'] = list(player.select("td"))[1].text.strip().replace(",", "").replace("'", "''")
                                player_te['te_url'] = list(player.select("a"))[0]['href']
                                players_te.append(player_te)
                            except ValueError:
                                print(Back.RED + "Hi ha una excepció amb el mestre " + te_name[1] + " " + te_name[0] + Style.RESET_ALL)

            page += 1

        print("\n" + Back.BLUE + "  JUGADORS QUE FALTEN  ")
        print(Style.RESET_ALL)

        for atp_id in country_players:
            print(Back.YELLOW + Fore.BLACK + "Falta trobar el mestre " + players_db[atp_id]['name'] + "(" + atp_id + ")")

# Update
doUpdate = True

if doUpdate:
    print(Style.RESET_ALL)
    print(Fore.GREEN + Style.BRIGHT)

    for player_te in players_te:
        rankdates = []
        query = "SELECT player_rankdate FROM player_by_keyword WHERE player_keyword = '" + player_te['keyword'] + "'"
        ranks = session.execute(query)

        for rank in ranks:
            rankdates.append(rank.player_rankdate)

        for rankdate in rankdates:
            update = "UPDATE player_by_keyword "\
                     "SET player_te_name = '" + player_te['te_name'] + "', "\
                     "player_te_url = '" + player_te['te_url'] + "' "\
                     "WHERE player_keyword = '" + player_te['keyword'] + "' "\
                     "AND player_rankdate = '" + str(rankdate) + "'"

            print(update)
            session.execute(update)

# Close connections
session.shutdown()
cluster.shutdown()
