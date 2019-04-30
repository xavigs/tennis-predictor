from cassandra.cluster import Cluster
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from pprint import pprint
import sys
from colorama import init, Fore, Back, Style
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Variables
init() # Init colorama
rounds = ["1R", "2R", "3R", "R16", "QF", "SF", "F"]
categories = ["250", "500", "1000", "grandslam"]
surfaces = ["H", "C", "G", "I"]
games = []
odds = []
dates_train = []
tournaments_train = []
games_names_train = []

# Open connection
cluster = Cluster(["127.0.0.1"])
session = cluster.connect("beast")

# Get games from DB
'''
    Torneig - Ronda - Local - Edat - Rank - Race - RankMax - H2H - %Any - %AnySup - %SupCar - 3Mesos - PtsDef - Odd
'''
query = "SELECT COUNT(*) as num_games FROM game_train"
row = session.execute(query)
num_games = row[0].num_games
num_games_train = round(num_games * 0.8, 0)
index = 0
season = 2014

while season < 2016:
    for week in range(1, 45):
        query = "SELECT game_season, game_tournament, game_week, game_round, game_surface, game_category, game_sets, game_points, game_date, game_rank1, game_rank2, game_race1, game_race2, game_rankmax1, game_rankmax2, game_age1, game_age2, game_h2h1, game_h2h2, game_h2h_year1, game_h2h_year2, game_h2h_surface1, game_h2h_surface2, game_h2h_surface_year1, game_h2h_surface_year2, game_surface1, game_surface2, game_surface_year1, game_surface_year2, game_hand1, game_hand2, game_home1, game_home2, game_3months1, game_3months2, game_10streak1, game_10streak2, game_gp1m1, game_gp1m2, game_gp3m1, game_gp3m2, game_gp6m1, game_gp6m2, game_pts_def1, game_pts_def2, game_player1, game_player2, game_winner, game_odd1, game_odd2, game_result FROM game_train WHERE game_season = " + str(season) + " AND game_week = " + str(week) + " ORDER BY game_id ASC"
        games_db = session.execute(query)

        for game_db in games_db:
            game = []
            game.append(round((rounds.index(game_db.game_round) + 1) / 7, 2))
            game.append(round((surfaces.index(game_db.game_surface) + 1) / 4, 2))
            game.append(round((categories.index(game_db.game_category) + 1) / 4, 2))

            # Ranking
            if game_db.game_rank1 == 0 or game_db.game_rank2 == 0:
                game.append(0)
                game.append(0)
            else:
                dif_rank_val = game_db.game_rank1 - game_db.game_rank2

                if 1 <= abs(dif_rank_val) <= 15:
                    dif_rank_val = -0.2 if dif_rank_val < 0 else 0.2
                elif 16 <= abs(dif_rank_val) <= 30:
                    dif_rank_val = -0.4 if dif_rank_val < 0 else 0.4
                elif 31 <= abs(dif_rank_val) <= 55:
                    dif_rank_val = -0.6 if dif_rank_val < 0 else 0.6
                elif 56 <= abs(dif_rank_val) <= 85:
                    dif_rank_val = -0.8 if dif_rank_val < 0 else 0.8
                else:
                    dif_rank_val = -1 if dif_rank_val < 0 else 1

                dif_rank_pct = round((game_db.game_rank1 - game_db.game_rank2) / max(game_db.game_rank1, game_db.game_rank2), 2)
                game.append(dif_rank_val)
                game.append(dif_rank_pct)

            # Race
            if game_db.game_race1 == 0 or game_db.game_race2 == 0:
                game.append(0)
                game.append(0)
            else:
                dif_race_val = game_db.game_race1 - game_db.game_race2

                if 1 <= abs(dif_race_val) <= 40:
                    dif_race_val = -0.5 if dif_race_val < 0 else 0.5
                else:
                    dif_race_val = -1 if dif_race_val < 0 else 1

                dif_race_pct = round((game_db.game_race1 - game_db.game_race2) / max(game_db.game_race1, game_db.game_race2), 2)
                game.append(dif_race_val)
                game.append(dif_race_pct)

            # Highest Ranking
            if game_db.game_rankmax1 == 0 or game_db.game_rankmax2 == 0:
                game.append(0)
                game.append(0)
            else:
                dif_rankmax_val = game_db.game_rankmax1 - game_db.game_rankmax2

                if 1 <= abs(dif_rankmax_val) <= 15:
                    dif_rankmax_val = -0.5 if dif_rankmax_val < 0 else 0.5
                else:
                    dif_rankmax_val = -1 if dif_rankmax_val < 0 else 1

                dif_rankmax_pct = round((game_db.game_rankmax1 - game_db.game_rankmax2) / max(game_db.game_rankmax1, game_db.game_rankmax2), 2)
                game.append(dif_rankmax_val)
                game.append(dif_rankmax_pct)

            if game_db.game_h2h1 == -1 or game_db.game_h2h2 == -1 or game_db.game_h2h1 == game_db.game_h2h2:
                game.append(0)
            else:
                dif_h2h = round((game_db.game_h2h2 - game_db.game_h2h1) / 10, 0) * 0.1
                game.append(dif_h2h)

            if game_db.game_h2h_year1 == -1 or game_db.game_h2h_year2 == -1 or game_db.game_h2h_year1 == game_db.game_h2h_year2:
                game.append(0)
            else:
                if game_db.game_h2h_year1 > game_db.game_h2h_year2:
                    game.append(-1)
                else:
                    game.append(1)

            if game_db.game_h2h_surface1 == -1 or game_db.game_h2h_surface2 == -1 or game_db.game_h2h_surface1 == game_db.game_h2h_surface2:
                game.append(0)
            else:
                if game_db.game_h2h_surface1 > game_db.game_h2h_surface2:
                    game.append(-1)
                else:
                    game.append(1)

            if game_db.game_surface1 == -1 or game_db.game_surface2 == -1 or game_db.game_surface1 == game_db.game_surface2:
                game.append(0)
            else:
                dif_surface = game_db.game_surface2 - game_db.game_surface1

                if dif_surface > 20:
                    game.append(1)
                elif dif_surface < -20:
                    game.append(-1)
                else:
                    dif_surface = round(dif_surface / 4, 0) * 0.2
                    game.append(dif_surface)

            if game_db.game_surface_year1 == -1 or game_db.game_surface_year2 == -1 or game_db.game_surface_year1 == game_db.game_surface_year2:
                game.append(0)
            else:
                dif_surface_year = game_db.game_surface_year2 - game_db.game_surface_year1

                if dif_surface_year > 30:
                    game.append(1)
                elif dif_surface_year < -30:
                    game.append(-1)
                else:
                    dif_surface_year = round(dif_surface_year / 6, 0) * 0.2
                    game.append(dif_surface_year)

            evol_3months1 = round((game_db.game_3months1 - game_db.game_rank1) * 100 / max(game_db.game_3months1, game_db.game_rank1), 0)
            evol_3months2 = round((game_db.game_3months2 - game_db.game_rank2) * 100 / max(game_db.game_3months2, game_db.game_rank2), 0)

            if evol_3months1 == evol_3months2:
                game.append(0)
            else:
                evol_3months = evol_3months2 - evol_3months1

                if evol_3months > 50:
                    game.append(1)
                elif evol_3months < -50:
                    game.append(-1)
                else:
                    evol_3months = round(evol_3months / 25, 0) * 0.5
                    game.append(evol_3months)

            if game_db.game_10streak1 == game_db.game_10streak2:
                game.append(0)
            else:
                dif_10streak = game_db.game_10streak2 - game_db.game_10streak1

                if dif_10streak > 20:
                    game.append(1)
                elif dif_10streak < -20:
                    game.append(-1)
                else:
                    dif_10streak = -0.5 if dif_10streak < 0 else 0.5
                    game.append(dif_10streak)

            if game_db.game_pts_def1 == game_db.game_pts_def2:
                game.append(0)
            else:
                dif_pts_def = game_db.game_pts_def2 - game_db.game_pts_def1

                if 1 <= abs(dif_pts_def) <= 90:
                    dif_pts_def = -0.5 if dif_pts_def < 0 else 0.5
                else:
                    dif_pts_def = -1 if dif_pts_def < 0 else 1

                game.append(dif_pts_def)

            game.append(round(game_db.game_points / 2000, 4))
            odd1 = game_db.game_odd1
            odd2 = game_db.game_odd2

            query_player1 = "SELECT player_name FROM player_by_atpid WHERE player_atpwt_id = '" + game_db.game_player1 + "'"
            player1 = session.execute(query_player1)
            player1_name = player1[0].player_name
            query_player2 = "SELECT player_name FROM player_by_atpid WHERE player_atpwt_id = '" + game_db.game_player2 + "'"
            player2 = session.execute(query_player2)
            player2_name = player2[0].player_name

            # Turn players randomly
            turn = random.randint(0, 1)

            if turn == 1:
                game[3] *= -1
                game[4] *= -1
                game[5] *= -1
                game[6] *= -1
                game[7] *= -1
                game[8] *= -1
                game[9] *= -1
                game[10] *= -1
                game[11] *= -1
                game[12] *= -1
                game[13] *= -1
                game[14] *= -1
                game[15] *= -1
                game[16] *= -1
                odd3 = odd1
                odd1 = odd2
                odd2 = odd3

                aux_name = player1_name
                player1_name = player2_name
                player2_name = aux_name
                game.append(1) # Winner
            else:
                game.append(-1) # Winner

            if "RET" not in game_db.game_result:
                #game.append(player1_name + " - " + player2_name)
                odds.append(str(odd1) + "-" + str(odd2))

                if index < num_games:
                    games.append(game)
                    games_names_train.append(player1_name + " - " + player2_name)
                    dates_train.append(str(game_db.game_date))
                    tournaments_train.append(game_db.game_tournament.capitalize())
                else:
                    predict.append(game)
                    games_names_predict.append(player1_name + " - " + player2_name)
                    dates_predict.append(str(game_db.game_date))
                    tournaments_predict.append(game_db.game_tournament.capitalize())

            index += 1
    season += 1

# Convert arrays to NumPy ndarrays
np.set_printoptions(suppress=True)
games_train = np.array(games)

# Construct DataFrame
data_frame = pd.DataFrame(games_train, columns = ["round", "surface", "category", "rank_val", "rank_pct", "race_val", "race_pct", "rankmax_val", "rankmax_pct", "h2h", "h2h_year", "h2h_surface", "dif_surface", "dif_surface_year", "evol_3months", "dif_10streak", "pts_def", "points", "winner"])

# Plot
#colors = ["#0033FF", "#FF4000"]
'''
colors = ["#082038", "#CCFF00"]
table = pd.crosstab(data_frame.pts_def, data_frame.winner)
table.div(table.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, color=colors)
figure = plt.get_current_fig_manager()
figure.canvas.set_window_title("The Beast")
plt.title("Defended Points vs Win/Loss")
plt.xlabel("Defended Points")
plt.ylabel("Win/Loss")
plt.show()
'''

# Training & Prediction
cat_vars = ["round", "surface", "category", "rank_val", "rank_pct", "race_val", "race_pct", "rankmax_val", "rankmax_pct", "h2h", "h2h_year", "h2h_surface", "dif_surface", "dif_surface_year", "evol_3months", "dif_10streak", "pts_def", "points"]
for var in cat_vars:
    cat_list = "var" + "_" + var
    cat_list = pd.get_dummies(data_frame[var], prefix = var)
    data1 = data_frame.join(cat_list)
    data_frame = data1

cat_vars = ["round", "surface", "category", "rank_val", "rank_pct", "race_val", "race_pct", "rankmax_val", "rankmax_pct", "h2h", "h2h_year", "h2h_surface", "dif_surface", "dif_surface_year", "evol_3months", "dif_10streak", "pts_def", "points"]
data_vars = data_frame.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]
data_final = data_frame[to_keep]

X = data_final.loc[:, data_final.columns != "winner"]
y = data_final.loc[:, data_final.columns == "winner"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train.values.ravel())
Y_pred = logmodel.predict(X_test)
Y_prob = logmodel.predict_proba(X_test)

print(Style.BRIGHT + Fore.YELLOW + "Precisió de la Regressió Logística (dades d'entrenament):")
print(Style.NORMAL + Fore.WHITE + str(logmodel.score(X_train, y_train)))
print(Style.BRIGHT + Fore.YELLOW + "Precisió de la Regressió Logística (dades de test):")
print(Style.NORMAL + Fore.WHITE + str(logmodel.score(X_test, y_test)) + "\n")

# Show results
index = 0
units = 0.0
hits = 0
stake = 0
predicted_ids = list(X_test.index)

for probability in Y_prob:
    players = games_names_train[predicted_ids[index]].split(" - ")
    odds_game = odds[predicted_ids[index]].split("-")
    odd1 = float(odds_game[0])
    odd2 = float(odds_game[1])
    prob1 = probability[0]
    prob2 = probability[1]
    inv_odd1 = 1 / odd1
    inv_odd2 = 1 / odd2
    inv_tot = inv_odd1 + inv_odd2
    prob_bookmark1 = round(inv_odd1 * (100 - (inv_tot * 100 - 100)), 2)
    prob_bookmark2 = round(inv_odd2 * (100 - (inv_tot * 100 - 100)), 2)
    new_odd1 = round(100 / prob_bookmark1, 2)
    new_odd2 = round(100 / prob_bookmark2, 2)
    value1 = round(odd1 * prob1, 2)
    value2 = round(odd2 * prob2, 2)

    if value1 >= value2:
        prediction = -1
        odd = odd1
        value = value1
        player_predicted = players[0]
    else:
        prediction = 1
        odd = odd2
        value = value2
        player_predicted = players[1]

    if int(games[predicted_ids[index]][-1]) == -1:
        winner = players[0]
    else:
        winner = players[1]

    if 1.75 <= odd <= 2.00 and value >= 1.25:
        stake += 1

        if int(games[predicted_ids[index]][-1]) == prediction:
            hits += 1
            result = "W"
            units += odd - 1
        else:
            result = "L"
            units -= 1

        sys.stdout.write(Style.BRIGHT + Fore.RED + "▸ " + Fore.WHITE + games_names_train[predicted_ids[index]] + " • " + Fore.MAGENTA + "Predicció : " + Style.NORMAL + Fore.WHITE + player_predicted + " • " + Style.BRIGHT + Fore.MAGENTA + "Valor : " + Style.NORMAL + Fore.WHITE + str(value) + " • " + Style.BRIGHT + Fore.MAGENTA + "Resultat real : " + Style.NORMAL + Fore.WHITE + winner + " ")
        sys.stdout.write(Style.BRIGHT)

        if result == "W":
            sys.stdout.write(Fore.GREEN)
        else:
            sys.stdout.write(Fore.RED)

        sys.stdout.write(result + "\n")
        print(Style.BRIGHT + Fore.YELLOW + "Odd: " + Style.NORMAL + Fore.WHITE + str(odd) + "\n")

    index += 1

print(Style.BRIGHT + Fore.YELLOW + "Hits: " + Style.NORMAL + Fore.WHITE + str(hits))
print(Style.BRIGHT + Fore.YELLOW + "Stake: " + Style.NORMAL + Fore.WHITE + str(stake))
print(Style.BRIGHT + Fore.YELLOW + "Units: " + Style.NORMAL + Fore.WHITE + str(units))
yield_pct = round(units * 100 / stake, 2)
print(Style.BRIGHT + Fore.YELLOW + "Yield: " + Style.NORMAL + Fore.WHITE + str(yield_pct) + "%")
