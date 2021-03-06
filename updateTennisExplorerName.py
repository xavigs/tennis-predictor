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

# Variables
init() # Init colorama
countries = dict()
players_db = dict()
players_te = []
countries_pycountry = ["Bolivia, Plurinational State of", "Bosnia and Herzegovina", "Czechia", "Dominican Republic", "United Kingdom", "Iran, Islamic Republic of", "Macedonia, Republic of", "Moldova, Republic of", "Papua New Guinea", "South Africa", "Russian Federation", "Korea, Republic of", "Taiwan, Province of China", "Tunisia", "United States", "Venezuela, Bolivarian Republic of", "Viet Nam"]
countries_te = ["Bolivia", "Bosnia and Herzeg.", "Czech Republic", "Dominican Rep.", "Great Britain", "Iran", "Macedonia", "Moldavsko", "Papua N. Guinea", "RSA", "Russia", "South Korea", "Taipei (CHN)", "Tunis", "USA", "Venezuela", "Vietnam"]
abbr_pycountry = ["BGR", "BRB", "CHE", "CHL", "DEU", "DNK", "DZA", "GRC", "HRV", "IDN", "IRN", "KWT", "LVA", "MCO", "NGA", "NLD", "OMN", "PHL", "PRI", "PRT", "PRY", "SLV", "SVN", "TWN", "URY", "VNM", "ZAF", "ZWE"]
abbr_atp = ["BUL", "BAR", "SUI", "CHI", "GER", "DEN", "ALG", "GRE", "CRO", "INA", "IRI", "KUW", "LAT", "MON", "NGR", "NED", "OMA", "PHI", "PUR", "POR", "PAR", "ESA", "SLO", "TPE", "URU", "VIE", "RSA", "ZIM"]
page = 1

# Get players from DB
cluster = Cluster(["127.0.0.1"])
session = cluster.connect("beast")

query = "SELECT player_keyword, player_atpwt_id, player_name, player_country, player_rankdate, player_te_name, player_te_url FROM player_by_atpid"
players = session.execute(query)
num_players = 0

for player in players:
    player_db = dict()
    player_db['name'] = player.player_name
    player_db['country'] = player.player_country
    player_db['rankdate'] = str(player.player_rankdate)
    player_db['keyword'] = player.player_keyword

    if player.player_te_name == "BLANK" and player.player_atpwt_id not in players_db:
        num_players += 1
        players_db[player.player_atpwt_id] = player_db
        #print(str(num_players) + ". " + player.player_name + "(" + player.player_country + ") => " + player.player_keyword)

        if player.player_country in countries:
            countries[player.player_country] += 1
        else:
            countries[player.player_country] = 1

        #if player.player_country == "KUW":
            #print(player.player_name)
#exit()
'''
countries = sorted(countries.items(), key=lambda kv: kv[1], reverse=True)
for country, count in countries:
    print(country, count)

print("Nº Players => " + str(num_players))
print("Nº Countries => " + str(len(countries)))
#print(list(pycountry.countries))
exit()

country = "Benin"
country_url = "benin"
country_pycountry = pycountry.countries.get(name=country)

if country_pycountry is None:
    country_pycountry = pycountry.countries.get(name=utils.replaceMultiple2(country, countries_te, countries_pycountry))

print(country_pycountry.alpha_3)

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
    url = "https://www.tennisexplorer.com/list-players/?country=" + country_url + "&page=" + str(page) + "&order=rank"
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

                if atp_id and te_name[1] + " " + te_name[0] != "Mohamed Hassan":
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
doUpdate = False

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
'''
# Update players manually
atp_ids = []
te_names = []
te_urls = []
#1
atp_ids.append("CH68")
te_names.append("Childers Julian Allen")
te_urls.append("/player/childers/")

print(Fore.GREEN + Style.BRIGHT)

for index, atp_id in enumerate(atp_ids):
    rankdates = []
    query = "SELECT player_rankdate FROM player_by_atpid WHERE player_atpwt_id = '" + atp_id + "'"
    ranks = session.execute(query)

    for rank in ranks:
        rankdates.append(rank.player_rankdate)

    for rankdate in rankdates:
        update = "UPDATE player_by_atpid "\
                 "SET player_te_name = '" + te_names[index] + "', "\
                 "player_te_url = '" + te_urls[index] + "' "\
                 "WHERE player_atpwt_id = '" + atp_id + "' "\
                 "AND player_rankdate = '" + str(rankdate) + "'"

        print(update)
        session.execute(update)

# Close connections
session.shutdown()
cluster.shutdown()
