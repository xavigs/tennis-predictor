from cassandra.cluster import Cluster
from bs4 import BeautifulSoup
import re
import requests
from datetime import datetime, timedelta, date
from pprint import pprint
import sys
sys.path.append(r'C:\Users\d_mas\Developer\The Beast\lib')
import utils
from colorama import init, Fore, Back, Style
from colorclass import Color, Windows
from terminaltables import SingleTable

# Variables
Windows.enable(auto_colors=True, reset_atexit=True)  # Does nothing if not on Windows
init() # Init colorama
rounds = {"F": "Finals", "SF": "Semi-Finals", "QF": "Quarter-Finals", "R16": "Round of 16"}
points = {"challenger": {"F": 125, "SF": 75, "QF": 45, "R16": 25, "1R": 10},
        "250": {"F": 250, "SF": 150, "QF": 90, "R16": 45, "1R": 20}}
season = 2014
id = 1
tournaments = []

# Get players from DB
cluster = Cluster(["127.0.0.1"])
session = cluster.connect("beast")

# Get tournaments from DB
query = "SELECT tournament_keyword, tournament_atpwt_id, tournament_category, tournament_country, tournament_end, tournament_name, tournament_num_players, tournament_start, tournament_surface FROM tournament WHERE tournament_season = " + str(season)
tournaments_db = session.execute(query)

for tournament_db in tournaments_db:
    tournament = dict()
    tournament['keyword'] = tournament_db.tournament_keyword
    tournament['atpwt_id'] = tournament_db.tournament_atpwt_id
    tournament['category'] = tournament_db.tournament_category
    tournament['country'] = tournament_db.tournament_country
    tournament['end'] = tournament_db.tournament_end
    tournament['name'] = tournament_db.tournament_name
    tournament['num_players'] = tournament_db.tournament_num_players
    tournament['start'] = tournament_db.tournament_start
    tournament['surface'] = tournament_db.tournament_surface
    tournaments.append(tournament)

tournaments = sorted(tournaments, key = lambda i: (i['end']))

for tournament in tournaments:
    games = dict()
    end_tournament = False
    set_def_dates = False

    # Set first rounds names
    if tournament['num_players'] <= 32:
        extra_rounds = {"1R": "Round of 32"}
    elif tournament['num_players'] <= 64:
        extra_rounds = {"2R": "Round of 32", "1R": "Round of 64"}
    else:
        extra_rounds = {"3R": "Round of 32", "2R": "Round of 64", "1R": "Round of 128"}

    # Get Data from ATP World Tour
    url = "https://www.atptour.com/en/scores/archive/" + tournament['keyword'] + "/" + str(tournament['atpwt_id']) + "/" + str(season) + "/results"
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data, "html.parser")
    games_atp = list(soup.select("table.day-table"))[0].find_all("tr")

    for game_atp in games_atp:
        link = game_atp.select("a")

        if len(link) == 0:
            # It's a round title
            round_atp = game_atp.text.strip()

            if "Quali" in round_atp:
                end_tournament = True
            else:
                round = utils.searchDictionary(rounds, round_atp)

                if not round:
                    round = utils.searchDictionary(extra_rounds, round_atp)

                games[round] = []
        else:
            # It's a game
            if not end_tournament:
                game = dict()
                game['id'] = id
                game['season'] = season
                game['tournament'] = tournament['keyword']
                game['round'] = round
                game['surface'] = tournament['surface']
                game['category'] = tournament['category']
                game['country'] = tournament['country']

                if tournament['category'] != "grandslam":
                    game['sets'] = 3
                else:
                    game['sets'] = 5

                href_player1 = list(link)[0]['href'].split("/")
                game['player1'] = href_player1[4].upper()

                query = "SELECT player_te_name, player_te_url, player_keyword, player_atpwt_id FROM player_by_atpid WHERE player_atpwt_id = '" + game['player1'] + "'"
                rows = session.execute(query)

                for player1 in rows:
                    game['player1_te_name'] = player1.player_te_name
                    game['player1_te_url'] = player1.player_te_url
                    game['player1_keyword'] = player1.player_keyword
                    break

                href_player2 = list(link)[1]['href'].split("/")
                game['player2'] = href_player2[4].upper()

                query = "SELECT player_te_name, player_te_url, player_keyword, player_atpwt_id FROM player_by_atpid WHERE player_atpwt_id = '" + game['player2'] + "'"
                rows = session.execute(query)

                for player2 in rows:
                    game['player2_te_name'] = player2.player_te_name
                    game['player2_te_url'] = player2.player_te_url
                    game['player2_keyword'] = player2.player_keyword
                    break

                game['winner'] = game['player1']
                game['result'] = list(link)[2].text.strip()
                games[round].append(game)
                id += 1

    # Get Data from Tennis Explorer
    index = 0
    url = "https://www.tennisexplorer.com/" + tournament['keyword'] + "/" + str(season) + "/atp-men/"
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data, "html.parser")
    games_te = list(soup.select("table.result"))[0].find_all("tr", id=re.compile("^r"))

    for game_te in games_te:
        pts_def1 = -1
        pts_def2 = -1

        if index % 2 == 0:
            # Player 1
            href_player1 = list(game_te.select("a"))[0]['href']
            round = list(game_te.select("td"))[1].text
            day = list(game_te.select("td"))[0].text[:5].split(".")

            if int(day[1]) == 12:
                year = season - 1
            else:
                year = season

            date_cql = str(year) + "-" + day[1] + "-" + day[0]
            hour = list(game_te.select("td"))[0].text[6:11].split(":")
            game_date = datetime(year, int(day[1]), int(day[0]), int(hour[0]), int(hour[1]), 0)
            week = date(year, int(day[1]), int(day[0])).isocalendar()[1]
            odd1 = list(game_te.select("td"))[9].text
            odd2 = list(game_te.select("td"))[10].text
        else:
            # Player 2
            href_player2 = list(game_te.select("a"))[0]['href']
            game_index = utils.findGameByPlayers(games, round, href_player1, href_player2)

            if game_index > -1: # Validate players IDs
                games[round][game_index]['week'] = week
                games[round][game_index]['points'] = points[tournament['category']][round]
                timestamp = int(str(datetime.timestamp(game_date)).split(".")[0])
                games[round][game_index]['date'] = game_date.strftime("%Y-%m-%d %H:%M:%S")
                games[round][game_index]['odd1'] = odd1
                games[round][game_index]['odd2'] = odd2

                player1_query = "SELECT player_ranking, player_race, player_hand, player_birth, player_country FROM player_by_atpid WHERE player_atpwt_id = '" + games[round][game_index]['player1'] + "' AND player_rankdate < '" + date_cql + "' ORDER BY player_rankdate DESC LIMIT 1"
                result = session.execute(player1_query)
                player1_data = result[0]

                games[round][game_index]['rank1'] = player1_data.player_ranking
                games[round][game_index]['race1'] = player1_data.player_race
                games[round][game_index]['hand1'] = player1_data.player_hand
                games[round][game_index]['age1'] = utils.calcAge(str(player1_data.player_birth), str(datetime.fromtimestamp(timestamp))[:10], False)

                if player1_data.player_country == tournament['country']:
                    games[round][game_index]['home1'] = True
                else:
                    games[round][game_index]['home1'] = False

                player2_query = "SELECT player_ranking, player_race, player_hand, player_birth, player_country FROM player_by_atpid WHERE player_atpwt_id = '" + games[round][game_index]['player2'] + "' AND player_rankdate < '" + date_cql + "' ORDER BY player_rankdate DESC LIMIT 1"
                result = session.execute(player2_query)
                player2_data = result[0]

                games[round][game_index]['rank2'] = player2_data.player_ranking
                games[round][game_index]['race2'] = player2_data.player_race
                games[round][game_index]['hand2'] = player2_data.player_hand
                games[round][game_index]['age2'] = utils.calcAge(str(player2_data.player_birth), str(datetime.fromtimestamp(timestamp))[:10], False)

                if player2_data.player_country == tournament['country']:
                    games[round][game_index]['home2'] = True
                else:
                    games[round][game_index]['home2'] = False

                # Rankmax
                url = "http://www.atpworldtour.com/en/players/" + games[round][game_index]['player1_keyword'] + "/" + games[round][game_index]['player1'] + "/rankings-history?ajax=true"
                r = requests.get(url)
                data = r.text
                soup = BeautifulSoup(data, "html.parser")
                weeks = utils.BSReverse(soup.select("table.mega-table tbody tr"))
                max_rank = 9999

                for week_rankmax in weeks:
                    cells = list(week_rankmax.select("td"))
                    ranking = int(cells[1].text.strip().replace("T", ""))
                    monday = cells[0].text.strip().replace(".", "-").split("-")
                    monday = int(str(datetime.timestamp(datetime(int(monday[0]), int(monday[1]), int(monday[2])))).split(".")[0])

                    if monday > timestamp:
                        break

                    if ranking > 0 and ranking < max_rank:
                        max_rank = ranking

                games[round][game_index]['rankmax1'] = max_rank

                url = "http://www.atpworldtour.com/en/players/" + games[round][game_index]['player2_keyword'] + "/" + games[round][game_index]['player2'] + "/rankings-history?ajax=true"
                r = requests.get(url)
                data = r.text
                soup = BeautifulSoup(data, "html.parser")
                weeks = utils.BSReverse(soup.select("table.mega-table tbody tr"))
                max_rank = 9999

                for week_rankmax in weeks:
                    cells = list(week_rankmax.select("td"))
                    ranking = int(cells[1].text.strip().replace("T", ""))
                    monday = cells[0].text.strip().replace(".", "-").split("-")
                    monday = int(str(datetime.timestamp(datetime(int(monday[0]), int(monday[1]), int(monday[2])))).split(".")[0])

                    if monday > timestamp:
                        break

                    if ranking > 0 and ranking < max_rank:
                        max_rank = ranking

                games[round][game_index]['rankmax2'] = max_rank

                # Head 2 Head
                url = "https://www.tennisexplorer.com/mutual/" + games[round][game_index]['player1_te_url'].replace("/player/", "") + games[round][game_index]['player2_te_url'].replace("/player/", "")
                r = requests.get(url)
                data = r.text
                soup = BeautifulSoup(data, "html.parser")
                rows = list(soup.select("table.result"))[1].select("tbody tr")

                win1 = 0
                win2 = 0
                games_played = 0
                win_surface1 = 0
                win_surface2 = 0
                games_played_surface = 0
                win_year1 = 0
                win_year2 = 0
                games_played_year = 0
                win_surface_year1 = 0
                win_surface_year2 = 0
                games_played_surface_year = 0
                start_date = str(year - 1) + "-" + day[1] + "-" + day[0]
                current_found = False

                for index_h2h, row in enumerate(rows):
                    if index_h2h % 2 == 0:
                        year_precedent = int(list(row.select("td"))[0].text.strip())
                        tournament_precedent = list(row.select("td"))[1].text.strip()
                        keyword_precedent = list(list(row.select("td"))[1].select("a"))[0]['href'].split("/")[1]
                        round_precedent = list(row.select("td"))[10].text.strip()
                        isQuali = True if round_precedent[:2] == "Q-" else False
                        isITF = True if round_precedent[:6] == "Future" else False
                        isHopman = True if round_precedent[:6] == "Hopman" else False
                        isChallenger = "challenger" in tournament_precedent
                        isChampionship = "Championship" in tournament_precedent
                        isDavis = "Davis Cup" in tournament_precedent
                        isExhibition = "exh." in tournament_precedent

                        if (year_precedent < season or year_precedent == season and current_found) and year_precedent != None and not isChampionship and not isDavis and not isExhibition and not isHopman:
                            # Game OK
                            games_played += 1
                            surface_precedent = list(list(row.select("td"))[4].select("span"))[0]['title'].upper()[:1]
                            winner_precedent = list(row.select("td"))[2].text.strip()[:-1]
                            te_name1 = games[round][game_index]['player1_te_name'].replace(",", "")
                            te_name2 = games[round][game_index]['player2_te_name'].replace(",", "")

                            # W/L General
                            if winner_precedent in te_name1:
                                winner_precedent = 1
                                win1 += 1
                            elif winner_precedent in te_name2:
                                winner_precedent = 2
                                win2 += 1
                            else:
                                print(Style.BRIGHT + Fore.RED + "Sniffff el güina del java-gueim no és cap dels pleias :(; diuen que és l'inusual " + winner_precedent)

                            # W/L Surface
                            if surface_precedent == tournament['surface']:
                                games_played_surface += 1

                                if winner_precedent == 1:
                                    win_surface1 += 1
                                elif winner_precedent == 2:
                                    win_surface2 += 1

                            # W/L Year
                            if (season - 2) < year_precedent < (season + 1):
                                url_tournament = "http://www.tennisexplorer.com/" + keyword_precedent + "/" + str(year_precedent) + "/atp-men/"
                                r_precedent = requests.get(url_tournament)
                                data_precedent = r_precedent.text
                                soup_precedent = BeautifulSoup(data_precedent, "html.parser")
                                games_tournament = list(soup_precedent.select("table.result"))[0].find_all("tr", id=re.compile("^r"))

                                if len(games_tournament) == 0:
                                    games_tournament = list(soup_precedent.select("table.result"))[1].find_all("tr", id=re.compile("^r"))

                                num_row = 0

                                for game_tournament in games_tournament:
                                    if num_row % 2 == 0:
                                        date_precedent = list(game_tournament.select("td"))[0].text.strip()[:5].split(".")

                                        if int(date_precedent[1]) == 12:
                                            date_precedent = str(year_precedent - 1) + "-" + date_precedent[1] + "-" + date_precedent[0]
                                        else:
                                            date_precedent = str(year_precedent) + "-" + date_precedent[1] + "-" + date_precedent[0]

                                        first_player = list(list(game_tournament.select("td"))[2].select("a"))[0]['href']
                                    else:
                                        second_player = list(list(game_tournament.select("td"))[0].select("a"))[0]['href']

                                        if (first_player == games[round][game_index]['player1_te_url'] and second_player == games[round][game_index]['player2_te_url'] or first_player == games[round][game_index]['player2_te_url'] and second_player == games[round][game_index]['player1_te_url']) and start_date <= date_precedent < date_cql:
                                            # It's a last 52-week game!
                                            games_played_year += 1

                                            if winner_precedent == 1:
                                                win_year1 += 1
                                            elif winner_precedent == 2:
                                                win_year2 += 1

                                            # W/L Year/Surface
                                            if surface_precedent == tournament['surface']:
                                                games_played_surface_year += 1

                                                if winner_precedent == 1:
                                                    win_surface_year1 += 1
                                                elif winner_precedent == 2:
                                                    win_surface_year2 += 1

                                            break

                                    num_row += 1
                        else:
                            if year_precedent == season and keyword_precedent == tournament['keyword']:
                                current_found = True

                if games_played == 0:
                    games[round][game_index]['h2h1'] = -1
                    games[round][game_index]['h2h2'] = -1
                    games[round][game_index]['h2h_surface1'] = -1
                    games[round][game_index]['h2h_surface2'] = -1
                    games[round][game_index]['h2h_year1'] = -1
                    games[round][game_index]['h2h_year2'] = -1
                    games[round][game_index]['h2h_surface_year1'] = -1
                    games[round][game_index]['h2h_surface_year2'] = -1
                else:
                    games[round][game_index]['h2h1'] = utils.around(win1 * 100 / games_played, 1)
                    games[round][game_index]['h2h2'] = utils.around(win2 * 100 / games_played, 1)

                    if games_played_surface == 0:
                        games[round][game_index]['h2h_surface1'] = -1
                        games[round][game_index]['h2h_surface2'] = -1
                    else:
                        games[round][game_index]['h2h_surface1'] = utils.around(win_surface1 * 100 / games_played_surface, 1)
                        games[round][game_index]['h2h_surface2'] = utils.around(win_surface2 * 100 / games_played_surface, 1)

                    if games_played_year == 0:
                        games[round][game_index]['h2h_year1'] = -1
                        games[round][game_index]['h2h_year2'] = -1
                    else:
                        games[round][game_index]['h2h_year1'] = utils.around(win_year1 * 100 / games_played_year, 1)
                        games[round][game_index]['h2h_year2'] = utils.around(win_year2 * 100 / games_played_year, 1)

                    if games_played_surface_year == 0:
                        games[round][game_index]['h2h_surface_year1'] = -1
                        games[round][game_index]['h2h_surface_year2'] = -1
                    else:
                        games[round][game_index]['h2h_surface_year1'] = utils.around(win_surface_year1 * 100 / games_played_surface_year, 1)
                        games[round][game_index]['h2h_surface_year2'] = utils.around(win_surface_year2 * 100 / games_played_surface_year, 1)

                # Ranking 3 Months
                date_3m_timestamp = timestamp - 7776000
                date_3m = datetime.fromtimestamp(date_3m_timestamp)

                if date_3m.weekday() > 0:
                    date_3m_timestamp = date_3m_timestamp - 86400 * date_3m.weekday()
                    date_3m = datetime.fromtimestamp(date_3m_timestamp)

                player1_query = "SELECT player_ranking FROM player_by_atpid WHERE player_atpwt_id = '" + games[round][game_index]['player1'] + "' AND player_rankdate = '" + str(date_3m)[:10] + "'"
                result = session.execute(player1_query)
                games[round][game_index]['3months1'] = result[0].player_ranking

                player2_query = "SELECT player_ranking FROM player_by_atpid WHERE player_atpwt_id = '" + games[round][game_index]['player2'] + "' AND player_rankdate = '" + str(date_3m)[:10] + "'"
                result = session.execute(player2_query)
                games[round][game_index]['3months2'] = result[0].player_ranking

                # Surfaces
                urls = ["https://www.tennisexplorer.com" + games[round][game_index]['player1_te_url'] + "?annual=all",
                        "https://www.tennisexplorer.com" + games[round][game_index]['player2_te_url'] + "?annual=all"]

                for index_surface, url in enumerate(urls):
                    wins_career = 0
                    games_played_career = 0
                    wins_year = 0
                    games_played_year = 0
                    r = requests.get(url)
                    data = r.text
                    soup = BeautifulSoup(data, "html.parser")

                    # Previous years
                    surfaces_te = ["C", "H", "I", "G"]
                    surface_index = surfaces_te.index(tournament['surface']) + 2
                    career_years = utils.BSReverse(list(soup.select("table.result.balance"))[0].select("tbody tr"))

                    for career_year in career_years:
                        current_year = int(list(career_year.select("td"))[0].text.strip())

                        if current_year >= year:
                            break
                        else:
                            balance = list(career_year.select("td"))[surface_index].text.strip().split("/")

                            if len(balance) > 1:
                                wins_career += int(balance[0])
                                games_played_career += int(balance[0]) + int(balance[1])

                    # Last 52 weeks, Last X-games and Points defended
                    games_year = list(soup.select("div#matches-" + str(season) + "-1-data"))[0].select("tr")
                    date_1m_timestamp = timestamp - 2592000
                    date_1m = str(datetime.fromtimestamp(date_1m_timestamp))[:10]
                    date_3m = str(date_3m)[:10]
                    date_6m_timestamp = timestamp - 15552000
                    date_6m = str(datetime.fromtimestamp(date_6m_timestamp))[:10]
                    last10_wins = 0
                    last10_found = 0
                    played_1m = 0
                    played_3m = 0
                    played_6m = 0
                    january_arrived = False

                    # Get start date and end date by week and year
                    if not set_def_dates:
                        d = date(year - 1, 1, 1)

                        if d.weekday() <= 3:
                            d = d - timedelta(d.weekday())
                        else:
                            d = d + timedelta(7 - d.weekday())

                        dlt = timedelta(days = (week - 1) * 7)
                        dlt2 = timedelta(days = (week - 1) * 7 - 1)
                        start_def_points = d + dlt2
                        end_def_points = d + dlt + timedelta(days = 6)
                        set_def_dates = True

                    tournaments_def_points = []

                    # Get tournaments from previous year
                    query = "SELECT tournament_keyword, tournament_atpwt_id, tournament_category, tournament_country, tournament_end, tournament_name, tournament_num_players, tournament_start, tournament_surface FROM tournament WHERE tournament_season = " + str(season - 1)
                    tournaments_prev = []
                    tournaments_db = session.execute(query)

                    for tournament_db in tournaments_db:
                        tournament_prev = dict()
                        tournament_prev['keyword'] = tournament_db.tournament_keyword
                        tournament_prev['atpwt_id'] = tournament_db.tournament_atpwt_id
                        tournament_prev['category'] = tournament_db.tournament_category
                        tournament_prev['country'] = tournament_db.tournament_country
                        tournament_prev['end'] = tournament_db.tournament_end
                        tournament_prev['name'] = tournament_db.tournament_name
                        tournament_prev['num_players'] = tournament_db.tournament_num_players
                        tournament_prev['start'] = tournament_db.tournament_start
                        tournament_prev['surface'] = tournament_db.tournament_surface
                        tournaments_prev.append(tournament_prev)

                    tournaments_prev = sorted(tournaments_prev, key = lambda i: (i['end']))

                    for tournament_prev in tournaments_prev:
                        if str(start_def_points) <= str(tournament_prev['end']) <= str(end_def_points):
                            tournaments_def_points.append(tournament_prev['keyword'])

                    # Current year games
                    for game_year in games_year:
                        if "head" not in game_year['class'] and "flags" not in game_year['class']:
                            date_game = list(game_year.select("td"))[0].text.strip().split(".")
                            hasSurface = len(list(game_year.select("td"))[1].select("span")) > 0
                            winner = list(list(game_year.select("td"))[2].select("a"))[0]['href']

                            if int(date_game[1]) == 1 and not january_arrived:
                                january_arrived = True

                            if january_arrived and int(date_game[1]) == 12:
                                date_game = str(year - 1) + "-" + date_game[1] + "-" + date_game[0]
                            else:
                                date_game = str(year) + "-" + date_game[1] + "-" + date_game[0]

                            if date_game < date_cql and last10_found < 10:
                                last10_found += 1

                                if winner == games[round][game_index]['player' + str(index_surface + 1) + '_te_url']:
                                    last10_wins += 1

                            if date_cql > date_game >= date_6m:
                                played_6m += 1

                                if date_game >= date_3m:
                                    played_3m += 1

                                    if date_game >= date_1m:
                                        played_1m += 1

                            if hasSurface:
                                surface_game = list(list(game_year.select("td"))[1].select("span"))[0]['title'].capitalize()[0]

                                if surface_game == tournament['surface'] and start_date <= date_game < date_cql:
                                    games_played_career += 1
                                    games_played_year += 1

                                    if winner == games[round][game_index]['player' + str(index_surface + 1) + '_te_url']:
                                        wins_career += 1
                                        wins_year += 1

                    games_year = list(soup.select("div#matches-" + str(season - 1) + "-1-data"))[0].select("tr")
                    january_arrived = False

                    # Last year games
                    for game_year in games_year:
                        if "head" not in game_year['class'] and "flags" not in game_year['class']:
                            # It's a game
                            date_game = list(game_year.select("td"))[0].text.strip().split(".")
                            hasSurface = len(list(game_year.select("td"))[1].select("span")) > 0
                            winner = list(list(game_year.select("td"))[2].select("a"))[0]['href']

                            if int(date_game[1]) == 1 and not january_arrived:
                                january_arrived = True

                            if january_arrived and int(date_game[1]) == 12:
                                date_game = str(season - 2) + "-" + date_game[1] + "-" + date_game[0]
                            else:
                                date_game = str(season - 1) + "-" + date_game[1] + "-" + date_game[0]

                            if date_game < date_cql and last10_found < 10:
                                last10_found += 1

                                if winner == games[round][game_index]['player' + str(index_surface + 1) + '_te_url']:
                                    last10_wins += 1

                            if date_cql > date_game >= date_6m:
                                played_6m += 1

                                if date_game >= date_3m:
                                    played_3m += 1

                                    if date_game >= date_1m:
                                        played_1m += 1

                            if hasSurface:
                                surface_game = list(list(game_year.select("td"))[1].select("span"))[0]['title'].capitalize()[0]

                                if surface_game == tournament['surface'] and start_date <= date_game < date_cql:
                                    games_played_year += 1

                                    if winner == games[round][game_index]['player' + str(index_surface + 1) + '_te_url']:
                                        wins_year += 1
                        else:
                            # Tournament title
                            if "Futures" not in game_year.text.strip():
                                current_tournament_keyword = list(game_year.select("a"))[0]['href'].split("/")[1]

                                if current_tournament_keyword in tournaments_def_points:
                                    last_round = list(game_year.find_next_sibling().select("td"))[3].text.strip()
                                    round_found = False
                                    has_won_some_game = False

                                    for key, content in points[game['category']].items():
                                        if round_found:
                                            has_won_some_game = True
                                            last_round = key
                                            break

                                        if key == last_round:
                                            round_found = True

                                    if index_surface == 0:
                                        if has_won_some_game:
                                            pts_def1 = points[game['category']][last_round]
                                        else:
                                            pts_def1 = 0

                                        games[round][game_index]['pts_def1'] = int(pts_def1)
                                    else:
                                        if has_won_some_game:
                                            pts_def2 = points[game['category']][last_round]
                                        else:
                                            pts_def2 = 0

                                        games[round][game_index]['pts_def2'] = int(pts_def2)
                                elif "challenger" in current_tournament_keyword:
                                    last_game_date = list(game_year.find_next_sibling().select("td"))[0].text.strip().split(".")

                                    if january_arrived and int(last_game_date[1]) == 12:
                                        last_game_date = str(season - 2) + "-" + last_game_date[1] + "-" + last_game_date[0]
                                    else:
                                        last_game_date = str(season - 1) + "-" + last_game_date[1] + "-" + last_game_date[0]

                                    if str(start_def_points) <= last_game_date <= str(end_def_points):
                                        last_round = list(game_year.find_next_sibling().select("td"))[3].text.strip()

                                        if index_surface == 0:
                                            pts_def1 = points['challenger'][last_round]
                                            games[round][game_index]['pts_def1'] = int(pts_def1)
                                        else:
                                            pts_def2 = points['challenger'][last_round]
                                            games[round][game_index]['pts_def2'] = int(pts_def2)


                    games[round][game_index]['surface' + str(index_surface + 1)] = utils.around(wins_career * 100 / games_played_career, 1)
                    games[round][game_index]['surface_year' + str(index_surface + 1)] = utils.around(wins_year * 100 / games_played_year, 1)
                    games[round][game_index]['gp1m' + str(index_surface + 1)] = played_1m
                    games[round][game_index]['gp3m' + str(index_surface + 1)] = played_3m
                    games[round][game_index]['gp6m' + str(index_surface + 1)] = played_6m

                    if last10_found < 10:
                        print(Style.BRIGHT + Fore.RED + "Al tanto, l'intrèpid " + games[round][game_index]['player' + str(index_surface + 1) + "_te_name"] + " no ha jugat 10 partits!")
                    else:
                        games[round][game_index]['10streak' + str(index_surface + 1)] = utils.around(last10_wins * 10, 1)

                if pts_def1 == -1:
                    games[round][game_index]['pts_def1'] = 0
                if pts_def2 == -1:
                    games[round][game_index]['pts_def2'] = 0

        index += 1

    # Print and save games
    current_round = ""
    table_games = []
    hidden_fields = ["id", "season", "tournament", "round", "player1_te_name", "player2_te_name"]
    text_fields = ["tournament", "round", "surface", "category", "country", "player1", "player2", "winner", "result", "date"]
    not_db_fields = ["player1_keyword", "player2_keyword", "player1_te_name", "player2_te_name", "player1_te_url", "player2_te_url"]

    for round, item in games.items():
        if round != current_round:
            current_round = round
            table_games.append([Color('{bgblue}{autowhite}Round: ' + round + '{/autowhite}{/bgblue}'), '', '', '', '', '', '', '', ''])

        for game in item:
            insert = "INSERT INTO game_train ("
            fields = []
            values = []
            player1_name = game['player1_keyword'].split("-")[1].capitalize()
            player2_name = game['player2_keyword'].split("-")[1].capitalize()
            row_fields = [Color('{autored}▸ {/autored}{autogreen}' + player1_name + " - " + player2_name + '{/autogreen}')]
            row_values = ['']
            printed_fields = 1

            for field, value in game.items():
                if field not in not_db_fields:
                    fields.append("game_" + field)

                    if field in text_fields:
                        values.append("'" + value + "'")
                    else:
                        values.append(value)

                if field not in hidden_fields:
                    if printed_fields == 0:
                        row_fields = ['']
                        row_values = ['']
                        printed_fields += 1

                    row_fields.append(Color('{autored}' + field + '{/autored}'))
                    row_values.append(value)
                    printed_fields += 1

                    if printed_fields == 9:
                        table_games.append(row_fields)
                        table_games.append(row_values)
                        printed_fields = 0

            insert += ', '.join([str(i) for i in fields]) + ") VALUES (" + ', '.join([str(i) for i in values]) + ")"
            print(insert)
            session.execute(insert)

            if printed_fields > 0:
                while printed_fields < 9:
                    row_fields.append('')
                    row_values.append('')
                    printed_fields += 1

                table_games.append(row_fields)
                table_games.append(row_values)

    table_instance = SingleTable(table_games, Color('{autoyellow} ' + tournament['name'] + ' ' + str(season) + ' (' + str(tournament['num_players']) + ' players) {/autoyellow}'))
    table_instance.inner_heading_row_border = False
    table_instance.inner_row_border = True
    table_instance.justify_columns = {0: 'left', 1: 'center', 2: 'center', 3: 'center', 4: 'center', 5: 'center', 6: 'center', 7: 'center', 8: 'center'}
    print("\n" + table_instance.table)
    break

# Close connections
session.shutdown()
cluster.shutdown()
