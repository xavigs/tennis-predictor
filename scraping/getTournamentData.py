from cassandra.cluster import Cluster
from bs4 import BeautifulSoup
import requests
import pycountry

# Variables
tournaments_db = []
seasons = ["2013", "2014", "2015", "2016", "2017"]
countries_pycountry = ["Russian Federation", "United Kingdom"]
countries_atp = ["Russia", "Great Britain"]
abbr_pycountry = ["ARE", "BGR", "CHE", "CHL", "DEU", "HRV", "MCO", "MYS", "NLD", "PRT"]
abbr_atp = ["UAE", "BUL", "SUI", "CHI", "GER", "CRO", "MON", "MAS", "NED", "POR"]

# Functions
def addslashes(s):
    l = ["\\", '"', "'", "\0", ]
    for i in l:
        if i in s:
            s = s.replace(i, i+i)
    return s

def replaceMultiple(mainString, toBeReplaces, newString):
    # Iterate over the strings to be replaced
    for elem in toBeReplaces :
        # Check if string is in the main string
        if elem in mainString :
            # Replace the string
            mainString = mainString.replace(elem, newString)

    return  mainString

def replaceMultiple2(mainString, origTuple, newTuple):
    # Iterate over the strings to be replaced
    for index, elem in enumerate(origTuple) :
        # Check if string is in the main string
        if elem in mainString :
            # Replace the string
            mainString = mainString.replace(elem, newTuple[index])

    return  mainString

# Web scraping - Tournaments
for season in seasons:
    tournaments_season = []
    url = "https://www.atptour.com/en/scores/results-archive?year=" + season
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data, "html.parser")

    for tournament in soup.select("tr.tourney-result"):
        tournament_db = dict()
        tournament_db['season'] = season
        tournament_data = list(tournament.select("td"))
        tournament_info = list(tournament_data[2].select("span"))
        # Name
        tournament_db['name'] = addslashes(tournament_info[0].text.strip())

        try:
            # Logo
            logo = list(tournament_data[1].select("img"))[0]
            toReplace = ["/assets/atpwt/images/tournament/badges/categorystamps_", ".png", ".svg"]
            tournament_db['category'] = replaceMultiple(logo['src'], toReplace, "")
        except IndexError:
            print("El torneig " + tournament_db['name'] + " (" + tournament_db['season'] + ") no té logo")
            tournament_db['category'] = "NoCat"

        try:
            location = tournament_info[1].text.strip().split(", ")
            # City
            tournament_db['city'] = location[0]
            # Country
            country = pycountry.countries.get(name=location[1])

            if country is None:
                country = pycountry.countries.get(name=replaceMultiple2(location[1], countries_atp, countries_pycountry))

            tournament_db['country'] = replaceMultiple2(country.alpha_3, abbr_pycountry, abbr_atp)
            # Start date
            tournament_db['start'] = tournament_info[2].text.strip().replace(".", "-")
            # Nº of players
            tournament_db['num_players'] = list(tournament_data[3].select("span"))[0].text.strip()
            # Surface
            tournament_db['atmosphere'] = list(tournament_data[4].select("div[class=item-details]"))[0].text.strip()
            # Append
            tournaments_season.append(tournament_db)
        except AttributeError:
            print("No s'ha trobat el país " + location[1])

    # Surfaces
    index = 0

    for span in enumerate(soup.select("span.item-value")):
        if 4 <= len(span[1] .text.strip()) <= 5:
            # It's the surface span
            surface = span[1].text.strip()
            if surface != "Hard" or tournaments_season[index]['atmosphere'] != "Indoor":
                tournaments_season[index]['surface'] = surface[0]
            else:
                tournaments_season[index]['surface'] = "I"

            index += 1

    # ATP Id + Keyword
    index = 0

    for link in enumerate(soup.select("td.tourney-details a")):
        if isinstance(link[1].get('href'), str) and "results" in link[1].get('href'):
            # It's the results link
            url = link[1].get('href').split("/")
            tournaments_season[index]['keyword'] = url[4]
            tournaments_season[index]['atpwt_id'] = url[5]
            index += 1

    # Join tuples
    tournaments_db += tournaments_season

# Insert
cluster = Cluster(["127.0.0.1"])
session = cluster.connect("beast")

for tournament_db in tournaments_db:
    if tournament_db['category'] != "NoCat":
        insert = "INSERT INTO tournament (tournament_season, tournament_atpwt_id, tournament_keyword, tournament_name, tournament_city, tournament_country, tournament_category, tournament_surface, tournament_start, tournament_end, tournament_num_players) VALUES (" + tournament_db['season'] + ", " + tournament_db['atpwt_id'] + ", '" + tournament_db['keyword'] + "', '" + tournament_db['name'] + "', '" + tournament_db['city'] + "', '" + tournament_db['country'] + "', '" + tournament_db['category'] + "', '" + tournament_db['surface'] + "', '" + tournament_db['start'] + "', '" + tournament_db['start'] + "', " + tournament_db['num_players'] + ")"
        print(insert)
        session.execute(insert)

# Close connections
session.shutdown()
cluster.shutdown()
