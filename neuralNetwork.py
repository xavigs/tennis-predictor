from cassandra.cluster import Cluster
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from colorclass import Color, Windows
from terminaltables import SingleTable
import random

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - x ** 2

class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_derivative

        # Init weights
        self.weights = []
        self.deltas = []

        # Set random values
        for i in range(1, len(layers) - 1):
            r = 2 * np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)

        r = 2 * np.random.random((layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        origin = X
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                dot_value = np.dot(a[l], self.weights[l])
                activation = self.activation(dot_value)
                a.append(activation)

            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            self.deltas.append(deltas)
            deltas.reverse()

            # Backpropagation
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 10000 == 0: print('epochs:', k)

            # Generem el valor del gràfic
            if k % (epochs / 200) == 0:
                global values
                total_errors = 0
                total_items = len(origin)

                for i, item in enumerate(origin):
                    coef = self.predict(item)

                    if coef[0] < 0 and y[i] == 1 or coef[0] > 0 and y[i] == -1:
                        total_errors += 1

                pct_errors = round((total_errors * 100) / total_items, 2)
                values.append(pct_errors)

                # Predicció
                global Z
                global sol
                global values_predicted
                total_errors = 0
                total_items = len(Z)

                for i, item in enumerate(Z):
                    coef = self.predict(item)

                    if coef[0] < 0 and sol[i] == 1 or coef[0] > 0 and sol[i] == -1:
                        total_errors += 1

                pct_errors = round((total_errors * 100) / total_items, 2)
                values_predicted.append(pct_errors)

    def predict(self, x):
        ones = np.atleast_2d(np.ones(x.shape[0]))
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    def print_weights(self):
        print("Llistat de pesos de connexions")
        for i in range(len(self.weights)):
            print(self.weights[i])

    def get_deltas(self):
        return self.deltas

# Variables
Windows.enable(auto_colors=True, reset_atexit=True)  # Does nothing if not on Windows
rounds = ["1R", "2R", "3R", "R16", "QF", "SF", "F"]
categories = ["250", "500", "1000", "grandslam"]
surfaces = ["H", "C", "G", "I"]
intervals = {"<1.50": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "1.50-1.99": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "2.00-2.49": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "2.50-2.99": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            ">2.99": {'units': 0.0, 'stake': 0, 'yield': 0.0}}
value_intervals = {"<1.01": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "1.01-1.50": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "1.51-2.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "2.01-2.50": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "2.51-3.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            ">3.00": {'units': 0.0, 'stake': 0, 'yield': 0.0}}

games_train = []
games = []
winners_train = []
games_predict = []
predict = []
winners_sim = []
dates_train = []
dates_predict = []
tournaments_train = []
tournaments_predict = []
games_names_train = []
games_names_predict = []
values = []
values_predicted = []
num_epochs = 50000000
nn = NeuralNetwork([21, 15, 15, 15, 1], activation = 'tanh')

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
        query = "SELECT game_season, game_tournament, game_week, game_round, game_surface, game_category, game_sets, game_points, game_date, game_rank1, game_rank2, game_race1, game_race2, game_rankmax1, game_rankmax2, game_age1, game_age2, game_h2h1, game_h2h2, game_h2h_year1, game_h2h_year2, game_h2h_surface1, game_h2h_surface2, game_h2h_surface_year1, game_h2h_surface_year2, game_surface1, game_surface2, game_surface_year1, game_surface_year2, game_hand1, game_hand2, game_home1, game_home2, game_3months1, game_3months2, game_10streak1, game_10streak2, game_gp1m1, game_gp1m2, game_gp3m1, game_gp3m2, game_gp6m1, game_gp6m2, game_pts_def1, game_pts_def2, game_player1, game_player2, game_winner, game_odd1, game_odd2 FROM game_train WHERE game_season = " + str(season) + " AND game_week = " + str(week) + " ORDER BY game_id ASC"
        games_db = session.execute(query)

        for game_db in games_db:
            game = []
            game.append(round((rounds.index(game_db.game_round) + 1) / 7, 2))
            game.append(round((surfaces.index(game_db.game_surface) + 1) / 4, 2))
            game.append(round((categories.index(game_db.game_category) + 1) / 4, 2))

            if game_db.game_rank1 == 0 or game_db.game_rank2 == 0:
                game.append(0)
            else:
                if game_db.game_rank1 < game_db.game_rank2:
                    game.append(-1)
                else:
                    game.append(1)

            if game_db.game_race1 == 0 or game_db.game_race2 == 0:
                game.append(0)
            else:
                if game_db.game_race1 < game_db.game_race2:
                    game.append(-1)
                else:
                    game.append(1)

            if game_db.game_rankmax1 == 0 or game_db.game_rankmax2 == 0 or game_db.game_rankmax1 == game_db.game_rankmax2:
                game.append(0)
            else:
                if game_db.game_rankmax1 < game_db.game_rankmax2:
                    game.append(-1)
                else:
                    game.append(1)

            if game_db.game_h2h1 == -1 or game_db.game_h2h2 == -1:
                game.append(0)
            else:
                if game_db.game_h2h1 > game_db.game_h2h2:
                    game.append(-1)
                else:
                    game.append(1)

            if game_db.game_h2h_year1 == -1 or game_db.game_h2h_year2 == -1:
                game.append(0)
            else:
                if game_db.game_h2h_year1 > game_db.game_h2h_year2:
                    game.append(-1)
                else:
                    game.append(1)

            if game_db.game_h2h_surface1 == -1 or game_db.game_h2h_surface2 == -1:
                game.append(0)
            else:
                if game_db.game_h2h_surface1 > game_db.game_h2h_surface2:
                    game.append(-1)
                else:
                    game.append(1)

            if game_db.game_surface1 == -1 or game_db.game_surface2 == -1 or game_db.game_surface1 == game_db.game_surface2:
                game.append(0)
            else:
                if game_db.game_surface1 > game_db.game_surface2:
                    game.append(-1)
                else:
                    game.append(1)

            if game_db.game_surface_year1 == -1 or game_db.game_surface_year2 == -1 or game_db.game_surface_year1 == game_db.game_surface_year2:
                game.append(0)
            else:
                if game_db.game_surface_year1 > game_db.game_surface_year2:
                    game.append(-1)
                else:
                    game.append(1)

            if game_db.game_home1 == game_db.game_home2:
                game.append(0)
            else:
                if game_db.game_home1:
                    game.append(-1)
                else:
                    game.append(1)

            evol_3months1 = round((game_db.game_3months1 - game_db.game_rank1) * 100 / max(game_db.game_3months1, game_db.game_rank1), 0)
            evol_3months2 = round((game_db.game_3months2 - game_db.game_rank2) * 100 / max(game_db.game_3months2, game_db.game_rank2), 0)

            if evol_3months1 == evol_3months2:
                game.append(0)
            else:
                if evol_3months1 > evol_3months2:
                    game.append(-1)
                else:
                    game.append(1)

            if game_db.game_10streak1 == game_db.game_10streak2:
                game.append(0)
            else:
                if game_db.game_10streak1 > game_db.game_10streak2:
                    game.append(-1)
                else:
                    game.append(1)

            if game_db.game_gp1m1 == game_db.game_gp1m2:
                game.append(0)
            else:
                if game_db.game_gp1m1 > game_db.game_gp1m2:
                    game.append(-1)
                else:
                    game.append(1)

            if game_db.game_gp3m1 == game_db.game_gp3m2:
                game.append(0)
            else:
                if game_db.game_gp3m1 > game_db.game_gp3m2:
                    game.append(-1)
                else:
                    game.append(1)

            if game_db.game_gp6m1 == game_db.game_gp6m2:
                game.append(0)
            else:
                if game_db.game_gp6m1 > game_db.game_gp6m2:
                    game.append(-1)
                else:
                    game.append(1)

            if game_db.game_pts_def1 == game_db.game_pts_def2:
                game.append(0)
            else:
                if game_db.game_pts_def1 > game_db.game_pts_def2:
                    game.append(-1)
                else:
                    game.append(1)

            game.append(round(game_db.game_points / 2000, 4))
            game.append(round(game_db.game_odd1 / 50, 4))
            game.append(round(game_db.game_odd2 / 50, 4))

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
                game[17] *= -1
                odd1 = game[19]
                game[19] = game[20]
                game[20] = odd1
                aux_name = player1_name
                player1_name = player2_name
                player2_name = aux_name
                winner = 1
            else:
                winner = -1

            if index < num_games_train:
                full_game = game.copy()
                games.append(full_game)
                #del game[0]
                games_train.append(game)
                games_names_train.append(player1_name + " - " + player2_name)
                dates_train.append(str(game_db.game_date))
                tournaments_train.append(game_db.game_tournament.capitalize())
                winners_train.append([winner])
            else:
                full_game = game.copy()
                predict.append(full_game)
                #del game[0]
                games_predict.append(game)
                games_names_predict.append(player1_name + " - " + player2_name)
                dates_predict.append(str(game_db.game_date))
                tournaments_predict.append(game_db.game_tournament.capitalize())
                winners_sim.append([winner])

            index += 1
    season += 1

np.set_printoptions(suppress=True)
X = np.array(games_train)
y = np.array(winners_train)
Z = np.array(games_predict)
sol = np.array(winners_sim)

# Previous prediction
index = 0

for e in X:
    print("X:", games_names_train[index], "Sol:", y[index], "Network:", nn.predict(e))
    index += 1

nn.fit(X, y, learning_rate = 0.01, epochs = num_epochs)

# Prediction after training
index = 0
train_table = []
train_table.append([Color('{autoyellow}Game{/autoyellow}'),
                            Color('{autoyellow}Date{/autoyellow}'),
                            Color('{autoyellow}Tournament{/autoyellow}'),
                            Color('{autoyellow}Round{/autoyellow}'),
                            Color('{autoyellow}Odd 1{/autoyellow}'),
                            Color('{autoyellow}Odd 2{/autoyellow}'),
                            Color('{autoyellow}Pick{/autoyellow}'),
                            Color('{autoyellow}Prob{/autoyellow}'),
                            Color('{autoyellow}Value{/autoyellow}'),
                            Color('{autoyellow}Res{/autoyellow}'),
                            Color('{autoyellow}Uts{/autoyellow}'),
                            Color('{autoyellow}Coef{/autoyellow}')])
units = 0.0
stake = 0
hits = 0
different_probs = []

for e in X:
    stake += 1
    row = []
    row.append(games_names_train[index])
    players = games_names_train[index].split(" - ")
    row.append(dates_train[index][:10])
    row.append(tournaments_train[index])
    row.append(rounds[int(round(games[index][0] * 6, 0))])
    odd1 = round(games[index][19] * 50, 2)
    odd2 = round(games[index][20] * 50, 2)
    row.append(odd1)
    row.append(odd2)

    coef = nn.predict(e)

    if round(coef[0], 2) not in different_probs:
        different_probs.append(round(coef[0], 2))

    prob2 = round((coef[0] + 1) * 100 / 2, 2)
    prob1 = round(100 - prob2, 2)
    inv_odd1 = 1 / odd1
    inv_odd2 = 1 / odd2
    inv_tot = inv_odd1 + inv_odd2
    prob_bookmark1 = round(inv_odd1 * (100 - (inv_tot * 100 - 100)), 2)
    prob_bookmark2 = round(inv_odd2 * (100 - (inv_tot * 100 - 100)), 2)
    new_odd1 = round(100 / prob_bookmark1, 2)
    new_odd2 = round(100 / prob_bookmark2, 2)
    value1 = round(odd1 * prob1 / 100, 2)
    value2 = round(odd2 * prob2 / 100, 2)

    if value1 >= value2:
        units_pick = odd1 - 1
        row.append(players[0])
        row.append(str(prob1) + "%")
        row.append(str(value1))

        if y[index] == -1:
            result = "W"
        else:
            result = "L"
    else:
        units_pick = odd2 - 1
        row.append(players[1])
        row.append(str(prob2) + "%")
        row.append(str(value2))

        if y[index] == -1:
            result = "L"
        else:
            result = "W"

    if result == "W":
        hits += 1
        row.append(Color('{autogreen}W{/autogreen}'))
        row.append("+" + str(round(units_pick, 2)))
        units += round(units_pick, 2)

        # Odds intervals
        if units_pick < 1.50:
            intervals['<1.50']['units'] += round(units_pick, 2)
            intervals['<1.50']['stake'] += 1
            intervals['<1.50']['yield'] = round(intervals['<1.50']['units'] * 100 / intervals['<1.50']['stake'], 2)
        elif units_pick < 2.00:
            intervals['1.50-1.99']['units'] += round(units_pick, 2)
            intervals['1.50-1.99']['stake'] += 1
            intervals['1.50-1.99']['yield'] = round(intervals['1.50-1.99']['units'] * 100 / intervals['1.50-1.99']['stake'], 2)
        elif units_pick < 2.49:
            intervals['2.00-2.49']['units'] += round(units_pick, 2)
            intervals['2.00-2.49']['stake'] += 1
            intervals['2.00-2.49']['yield'] = round(intervals['2.00-2.49']['units'] * 100 / intervals['2.00-2.49']['stake'], 2)
        elif units_pick < 2.99:
            intervals['2.50-2.99']['units'] += round(units_pick, 2)
            intervals['2.50-2.99']['stake'] += 1
            intervals['2.50-2.99']['yield'] = round(intervals['2.50-2.99']['units'] * 100 / intervals['2.50-2.99']['stake'], 2)
        else:
            intervals['>2.99']['units'] += round(units_pick, 2)
            intervals['>2.99']['stake'] += 1
            intervals['>2.99']['yield'] = round(intervals['>2.99']['units'] * 100 / intervals['>2.99']['stake'], 2)

        # Value intervals
        if float(row[8]) < 1.01:
            value_intervals['<1.01']['units'] += round(units_pick, 2)
            value_intervals['<1.01']['stake'] += 1
            value_intervals['<1.01']['yield'] = round(value_intervals['<1.01']['units'] * 100 / value_intervals['<1.01']['stake'], 2)
        elif float(row[8]) < 1.51:
            value_intervals['1.01-1.50']['units'] += round(units_pick, 2)
            value_intervals['1.01-1.50']['stake'] += 1
            value_intervals['1.01-1.50']['yield'] = round(value_intervals['1.01-1.50']['units'] * 100 / value_intervals['1.01-1.50']['stake'], 2)
        elif float(row[8]) < 2.01:
            value_intervals['1.51-2.00']['units'] += round(units_pick, 2)
            value_intervals['1.51-2.00']['stake'] += 1
            value_intervals['1.51-2.00']['yield'] = round(value_intervals['1.51-2.00']['units'] * 100 / value_intervals['1.51-2.00']['stake'], 2)
        elif float(row[8]) < 2.51:
            value_intervals['2.01-2.50']['units'] += round(units_pick, 2)
            value_intervals['2.01-2.50']['stake'] += 1
            value_intervals['2.01-2.50']['yield'] = round(value_intervals['2.01-2.50']['units'] * 100 / value_intervals['2.01-2.50']['stake'], 2)
        elif float(row[8]) < 3.01:
            value_intervals['2.51-3.00']['units'] += round(units_pick, 2)
            value_intervals['2.51-3.00']['stake'] += 1
            value_intervals['2.51-3.00']['yield'] = round(value_intervals['2.51-3.00']['units'] * 100 / value_intervals['2.51-3.00']['stake'], 2)
        else:
            value_intervals['>3.00']['units'] += round(units_pick, 2)
            value_intervals['>3.00']['stake'] += 1
            value_intervals['>3.00']['yield'] = round(value_intervals['>3.00']['units'] * 100 / value_intervals['>3.00']['stake'], 2)
    else:
        row.append(Color('{autored}L{/autored}'))
        row.append(-1)
        units -= 1

        # Odds intervals
        if units_pick < 1.50:
            intervals['<1.50']['units'] -= 1
            intervals['<1.50']['stake'] += 1
            intervals['<1.50']['yield'] = round(intervals['<1.50']['units'] * 100 / intervals['<1.50']['stake'], 2)
        elif units_pick < 2.00:
            intervals['1.50-1.99']['units'] -= 1
            intervals['1.50-1.99']['stake'] += 1
            intervals['1.50-1.99']['yield'] = round(intervals['1.50-1.99']['units'] * 100 / intervals['1.50-1.99']['stake'], 2)
        elif units_pick < 2.49:
            intervals['2.00-2.49']['units'] -= 1
            intervals['2.00-2.49']['stake'] += 1
            intervals['2.00-2.49']['yield'] = round(intervals['2.00-2.49']['units'] * 100 / intervals['2.00-2.49']['stake'], 2)
        elif units_pick < 2.99:
            intervals['2.50-2.99']['units'] -= 1
            intervals['2.50-2.99']['stake'] += 1
            intervals['2.50-2.99']['yield'] = round(intervals['2.50-2.99']['units'] * 100 / intervals['2.50-2.99']['stake'], 2)
        else:
            intervals['>2.99']['units'] -= 1
            intervals['>2.99']['stake'] += 1
            intervals['>2.99']['yield'] = round(intervals['>2.99']['units'] * 100 / intervals['>2.99']['stake'], 2)

        # Value intervals
        if float(row[8]) < 1.01:
            value_intervals['<1.01']['units'] -= 1
            value_intervals['<1.01']['stake'] += 1
            value_intervals['<1.01']['yield'] = round(value_intervals['<1.01']['units'] * 100 / value_intervals['<1.01']['stake'], 2)
        elif float(row[8]) < 1.51:
            value_intervals['1.01-1.50']['units'] -= 1
            value_intervals['1.01-1.50']['stake'] += 1
            value_intervals['1.01-1.50']['yield'] = round(value_intervals['1.01-1.50']['units'] * 100 / value_intervals['1.01-1.50']['stake'], 2)
        elif float(row[8]) < 2.01:
            value_intervals['1.51-2.00']['units'] -= 1
            value_intervals['1.51-2.00']['stake'] += 1
            value_intervals['1.51-2.00']['yield'] = round(value_intervals['1.51-2.00']['units'] * 100 / value_intervals['1.51-2.00']['stake'], 2)
        elif float(row[8]) < 2.51:
            value_intervals['2.01-2.50']['units'] -= 1
            value_intervals['2.01-2.50']['stake'] += 1
            value_intervals['2.01-2.50']['yield'] = round(value_intervals['2.01-2.50']['units'] * 100 / value_intervals['2.01-2.50']['stake'], 2)
        elif float(row[8]) < 3.01:
            value_intervals['2.51-3.00']['units'] -= 1
            value_intervals['2.51-3.00']['stake'] += 1
            value_intervals['2.51-3.00']['yield'] = round(value_intervals['2.51-3.00']['units'] * 100 / value_intervals['2.51-3.00']['stake'], 2)
        else:
            value_intervals['>3.00']['units'] -= 1
            value_intervals['>3.00']['stake'] += 1
            value_intervals['>3.00']['yield'] = round(value_intervals['>3.00']['units'] * 100 / value_intervals['>3.00']['stake'], 2)

    row.append(coef[0])
    train_table.append(row)
    index += 1

table_instance = SingleTable(train_table, Color('{autocyan} Prediction with training games {/autocyan}'))
table_instance.inner_heading_row_border = False
table_instance.inner_row_border = True
table_instance.justify_columns = {0: 'left', 1: 'center', 2: 'center', 3: 'center', 4: 'center', 5: 'center', 6: 'left', 7: 'center', 8: 'center', 9: 'center', 10: 'center', 11: 'center'}
print("\n" + table_instance.table)

print("\n" + Color('{autogreen}Units: {/autogreen}') + str(round(units, 2)))
print(Color('{autogreen}Hits: {/autogreen}') + str(hits))
print(Color('{autogreen}Stake: {/autogreen}') + str(stake))
print(Color('{autogreen}Yield: {/autogreen}') + str(round(units * 100 / stake, 2)) + "%")
print(Color('{autogreen}Different probabilities: {/autogreen}') + str(len(different_probs)))
print(Color('{autogreen}Value intervals: {/autogreen}'))
pprint(value_intervals)
print(Color('{autogreen}Odd intervals: {/autogreen}'))
pprint(intervals)

# Prediction with new games
index = 0
intervals = {"<1.50": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "1.50-1.99": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "2.00-2.49": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "2.50-2.99": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            ">2.99": {'units': 0.0, 'stake': 0, 'yield': 0.0}}
value_intervals = {"<1.01": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "1.01-1.50": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "1.51-2.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "2.01-2.50": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "2.51-3.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            ">3.00": {'units': 0.0, 'stake': 0, 'yield': 0.0}}
odd_value_intervals = {"Odd <1.50 / Value <1.01": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd <1.50 / Value 1.01-1.50": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd <1.50 / Value 1.51-2.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd <1.50 / Value 2.01-2.50": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd <1.50 / Value 2.51-3.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd <1.50 / Value >3.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 1.50-1.99 / Value <1.01": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 1.50-1.99 / Value 1.01-1.50": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 1.50-1.99 / Value 1.51-2.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 1.50-1.99 / Value 2.01-2.50": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 1.50-1.99 / Value 2.51-3.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 1.50-1.99 / Value >3.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 2.00-2.49 / Value <1.01": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 2.00-2.49 / Value 1.01-1.50": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 2.00-2.49 / Value 1.51-2.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 2.00-2.49 / Value 2.01-2.50": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 2.00-2.49 / Value 2.51-3.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 2.00-2.49 / Value >3.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 2.50-2.99 / Value <1.01": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 2.50-2.99 / Value 1.01-1.50": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 2.50-2.99 / Value 1.51-2.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 2.50-2.99 / Value 2.01-2.50": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 2.50-2.99 / Value 2.51-3.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd 2.50-2.99 / Value >3.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd >2.99 / Value <1.01": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd >2.99 / Value 1.01-1.50": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd >2.99 / Value 1.51-2.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd >2.99 / Value 2.01-2.50": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd >2.99 / Value 2.51-3.00": {'units': 0.0, 'stake': 0, 'yield': 0.0},
            "Odd >2.99 / Value >3.00": {'units': 0.0, 'stake': 0, 'yield': 0.0}}
prediction_table = []
prediction_table.append([Color('{autoyellow}Game{/autoyellow}'),
                            Color('{autoyellow}Date{/autoyellow}'),
                            Color('{autoyellow}Tournament{/autoyellow}'),
                            Color('{autoyellow}Round{/autoyellow}'),
                            Color('{autoyellow}Odd 1{/autoyellow}'),
                            Color('{autoyellow}Odd 2{/autoyellow}'),
                            Color('{autoyellow}Pick{/autoyellow}'),
                            Color('{autoyellow}Prob{/autoyellow}'),
                            Color('{autoyellow}Value{/autoyellow}'),
                            Color('{autoyellow}Res{/autoyellow}'),
                            Color('{autoyellow}Uts{/autoyellow}'),
                            Color('{autoyellow}Coef{/autoyellow}')])
units = 0.0
stake = 0
hits = 0
different_probs = []

for e in Z:
    row = []
    row.append(games_names_predict[index])
    players = games_names_predict[index].split(" - ")
    row.append(dates_predict[index][:10])
    row.append(tournaments_predict[index])
    row.append(rounds[int(round(predict[index][0] * 6, 0))])
    odd1 = round(predict[index][19] * 50, 2)
    odd2 = round(predict[index][20] * 50, 2)
    row.append(odd1)
    row.append(odd2)

    coef = nn.predict(e)

    if round(coef[0], 2) not in different_probs:
        different_probs.append(round(coef[0], 2))

    prob2 = round((coef[0] + 1) * 100 / 2, 2)
    prob1 = round(100 - prob2, 2)
    inv_odd1 = 1 / odd1
    inv_odd2 = 1 / odd2
    inv_tot = inv_odd1 + inv_odd2
    prob_bookmark1 = round(inv_odd1 * (100 - (inv_tot * 100 - 100)), 2)
    prob_bookmark2 = round(inv_odd2 * (100 - (inv_tot * 100 - 100)), 2)
    new_odd1 = round(100 / prob_bookmark1, 2)
    new_odd2 = round(100 / prob_bookmark2, 2)
    value1 = round(odd1 * prob1 / 100, 2)
    value2 = round(odd2 * prob2 / 100, 2)

    if value1 >= value2:
        units_pick = odd1 - 1
        row.append(players[0])
        row.append(str(prob1) + "%")
        row.append(str(value1))

        if sol[index] == -1:
            result = "W"
        else:
            result = "L"
    else:
        units_pick = odd2 - 1
        row.append(players[1])
        row.append(str(prob2) + "%")
        row.append(str(value2))

        if sol[index] == -1:
            result = "L"
        else:
            result = "W"

    if 1.50 < float(row[8]) <= 2.00 and 0.50 <= float(units_pick) < 2.00:
        stake += 1

        if result == "W":
            hits += 1
            row.append(Color('{autogreen}W{/autogreen}'))
            row.append("+" + str(round(units_pick, 2)))
            units += round(units_pick, 2)

            # Odds intervals
            if float(units_pick) < 0.50:
                intervals['<1.50']['units'] += round(units_pick, 2)
                intervals['<1.50']['stake'] += 1
                intervals['<1.50']['yield'] = round(intervals['<1.50']['units'] * 100 / intervals['<1.50']['stake'], 2)

                if float(row[8]) < 1.01:
                    odd_value_intervals['Odd <1.50 / Value <1.01']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd <1.50 / Value <1.01']['stake'] += 1
                    odd_value_intervals['Odd <1.50 / Value <1.01']['yield'] = round(odd_value_intervals['Odd <1.50 / Value <1.01']['units'] * 100 / odd_value_intervals['Odd <1.50 / Value <1.01']['stake'], 2)
                elif float(row[8]) < 1.51:
                    odd_value_intervals['Odd <1.50 / Value 1.01-1.50']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd <1.50 / Value 1.01-1.50']['stake'] += 1
                    odd_value_intervals['Odd <1.50 / Value 1.01-1.50']['yield'] = round(odd_value_intervals['Odd <1.50 / Value 1.01-1.50']['units'] * 100 / odd_value_intervals['Odd <1.50 / Value 1.01-1.50']['stake'], 2)
                elif float(row[8]) < 2.01:
                    odd_value_intervals['Odd <1.50 / Value 1.51-2.00']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd <1.50 / Value 1.51-2.00']['stake'] += 1
                    odd_value_intervals['Odd <1.50 / Value 1.51-2.00']['yield'] = round(odd_value_intervals['Odd <1.50 / Value 1.51-2.00']['units'] * 100 / odd_value_intervals['Odd <1.50 / Value 1.51-2.00']['stake'], 2)
                elif float(row[8]) < 2.51:
                    odd_value_intervals['Odd <1.50 / Value 2.01-2.50']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd <1.50 / Value 2.01-2.50']['stake'] += 1
                    odd_value_intervals['Odd <1.50 / Value 2.01-2.50']['yield'] = round(odd_value_intervals['Odd <1.50 / Value 2.01-2.50']['units'] * 100 / odd_value_intervals['Odd <1.50 / Value 2.01-2.50']['stake'], 2)
                elif float(row[8]) < 3.01:
                    odd_value_intervals['Odd <1.50 / Value 2.51-3.00']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd <1.50 / Value 2.51-3.00']['stake'] += 1
                    odd_value_intervals['Odd <1.50 / Value 2.51-3.00']['yield'] = round(odd_value_intervals['Odd <1.50 / Value 2.51-3.00']['units'] * 100 / odd_value_intervals['Odd <1.50 / Value 2.51-3.00']['stake'], 2)
                else:
                    odd_value_intervals['Odd <1.50 / Value >3.00']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd <1.50 / Value >3.00']['stake'] += 1
                    odd_value_intervals['Odd <1.50 / Value >3.00']['yield'] = round(odd_value_intervals['Odd <1.50 / Value >3.00']['units'] * 100 / odd_value_intervals['Odd <1.50 / Value >3.00']['stake'], 2)
            elif float(units_pick) < 1.00:
                intervals['1.50-1.99']['units'] += round(units_pick, 2)
                intervals['1.50-1.99']['stake'] += 1
                intervals['1.50-1.99']['yield'] = round(intervals['1.50-1.99']['units'] * 100 / intervals['1.50-1.99']['stake'], 2)

                if float(row[8]) < 1.01:
                    odd_value_intervals['Odd 1.50-1.99 / Value <1.01']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 1.50-1.99 / Value <1.01']['stake'] += 1
                    odd_value_intervals['Odd 1.50-1.99 / Value <1.01']['yield'] = round(odd_value_intervals['Odd 1.50-1.99 / Value <1.01']['units'] * 100 / odd_value_intervals['Odd 1.50-1.99 / Value <1.01']['stake'], 2)
                elif float(row[8]) < 1.51:
                    odd_value_intervals['Odd 1.50-1.99 / Value 1.01-1.50']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 1.50-1.99 / Value 1.01-1.50']['stake'] += 1
                    odd_value_intervals['Odd 1.50-1.99 / Value 1.01-1.50']['yield'] = round(odd_value_intervals['Odd 1.50-1.99 / Value 1.01-1.50']['units'] * 100 / odd_value_intervals['Odd 1.50-1.99 / Value 1.01-1.50']['stake'], 2)
                elif float(row[8]) < 2.01:
                    odd_value_intervals['Odd 1.50-1.99 / Value 1.51-2.00']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 1.50-1.99 / Value 1.51-2.00']['stake'] += 1
                    odd_value_intervals['Odd 1.50-1.99 / Value 1.51-2.00']['yield'] = round(odd_value_intervals['Odd 1.50-1.99 / Value 1.51-2.00']['units'] * 100 / odd_value_intervals['Odd 1.50-1.99 / Value 1.51-2.00']['stake'], 2)
                elif float(row[8]) < 2.51:
                    odd_value_intervals['Odd 1.50-1.99 / Value 2.01-2.50']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 1.50-1.99 / Value 2.01-2.50']['stake'] += 1
                    odd_value_intervals['Odd 1.50-1.99 / Value 2.01-2.50']['yield'] = round(odd_value_intervals['Odd 1.50-1.99 / Value 2.01-2.50']['units'] * 100 / odd_value_intervals['Odd 1.50-1.99 / Value 2.01-2.50']['stake'], 2)
                elif float(row[8]) < 3.01:
                    odd_value_intervals['Odd 1.50-1.99 / Value 2.51-3.00']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 1.50-1.99 / Value 2.51-3.00']['stake'] += 1
                    odd_value_intervals['Odd 1.50-1.99 / Value 2.51-3.00']['yield'] = round(odd_value_intervals['Odd 1.50-1.99 / Value 2.51-3.00']['units'] * 100 / odd_value_intervals['Odd 1.50-1.99 / Value 2.51-3.00']['stake'], 2)
                else:
                    odd_value_intervals['Odd 1.50-1.99 / Value >3.00']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 1.50-1.99 / Value >3.00']['stake'] += 1
                    odd_value_intervals['Odd 1.50-1.99 / Value >3.00']['yield'] = round(odd_value_intervals['Odd 1.50-1.99 / Value >3.00']['units'] * 100 / odd_value_intervals['Odd 1.50-1.99 / Value >3.00']['stake'], 2)
            elif float(units_pick) < 1.50:
                intervals['2.00-2.49']['units'] += round(units_pick, 2)
                intervals['2.00-2.49']['stake'] += 1
                intervals['2.00-2.49']['yield'] = round(intervals['2.00-2.49']['units'] * 100 / intervals['2.00-2.49']['stake'], 2)

                if float(row[8]) < 1.01:
                    odd_value_intervals['Odd 2.00-2.49 / Value <1.01']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 2.00-2.49 / Value <1.01']['stake'] += 1
                    odd_value_intervals['Odd 2.00-2.49 / Value <1.01']['yield'] = round(odd_value_intervals['Odd 2.00-2.49 / Value <1.01']['units'] * 100 / odd_value_intervals['Odd 2.00-2.49 / Value <1.01']['stake'], 2)
                elif float(row[8]) < 1.51:
                    odd_value_intervals['Odd 2.00-2.49 / Value 1.01-1.50']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 2.00-2.49 / Value 1.01-1.50']['stake'] += 1
                    odd_value_intervals['Odd 2.00-2.49 / Value 1.01-1.50']['yield'] = round(odd_value_intervals['Odd 2.00-2.49 / Value 1.01-1.50']['units'] * 100 / odd_value_intervals['Odd 2.00-2.49 / Value 1.01-1.50']['stake'], 2)
                elif float(row[8]) < 2.01:
                    odd_value_intervals['Odd 2.00-2.49 / Value 1.51-2.00']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 2.00-2.49 / Value 1.51-2.00']['stake'] += 1
                    odd_value_intervals['Odd 2.00-2.49 / Value 1.51-2.00']['yield'] = round(odd_value_intervals['Odd 2.00-2.49 / Value 1.51-2.00']['units'] * 100 / odd_value_intervals['Odd 2.00-2.49 / Value 1.51-2.00']['stake'], 2)
                elif float(row[8]) < 2.51:
                    odd_value_intervals['Odd 2.00-2.49 / Value 2.01-2.50']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 2.00-2.49 / Value 2.01-2.50']['stake'] += 1
                    odd_value_intervals['Odd 2.00-2.49 / Value 2.01-2.50']['yield'] = round(odd_value_intervals['Odd 2.00-2.49 / Value 2.01-2.50']['units'] * 100 / odd_value_intervals['Odd 2.00-2.49 / Value 2.01-2.50']['stake'], 2)
                elif float(row[8]) < 3.01:
                    odd_value_intervals['Odd 2.00-2.49 / Value 2.51-3.00']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 2.00-2.49 / Value 2.51-3.00']['stake'] += 1
                    odd_value_intervals['Odd 2.00-2.49 / Value 2.51-3.00']['yield'] = round(odd_value_intervals['Odd 2.00-2.49 / Value 2.51-3.00']['units'] * 100 / odd_value_intervals['Odd 2.00-2.49 / Value 2.51-3.00']['stake'], 2)
                else:
                    odd_value_intervals['Odd 2.00-2.49 / Value >3.00']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 2.00-2.49 / Value >3.00']['stake'] += 1
                    odd_value_intervals['Odd 2.00-2.49 / Value >3.00']['yield'] = round(odd_value_intervals['Odd 2.00-2.49 / Value >3.00']['units'] * 100 / odd_value_intervals['Odd 2.00-2.49 / Value >3.00']['stake'], 2)
            elif float(units_pick) < 2.00:
                intervals['2.50-2.99']['units'] += round(units_pick, 2)
                intervals['2.50-2.99']['stake'] += 1
                intervals['2.50-2.99']['yield'] = round(intervals['2.50-2.99']['units'] * 100 / intervals['2.50-2.99']['stake'], 2)

                if float(row[8]) < 1.01:
                    odd_value_intervals['Odd 2.50-2.99 / Value <1.01']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 2.50-2.99 / Value <1.01']['stake'] += 1
                    odd_value_intervals['Odd 2.50-2.99 / Value <1.01']['yield'] = round(odd_value_intervals['Odd 2.50-2.99 / Value <1.01']['units'] * 100 / odd_value_intervals['Odd 2.50-2.99 / Value <1.01']['stake'], 2)
                elif float(row[8]) < 1.51:
                    odd_value_intervals['Odd 2.50-2.99 / Value 1.01-1.50']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 2.50-2.99 / Value 1.01-1.50']['stake'] += 1
                    odd_value_intervals['Odd 2.50-2.99 / Value 1.01-1.50']['yield'] = round(odd_value_intervals['Odd 2.50-2.99 / Value 1.01-1.50']['units'] * 100 / odd_value_intervals['Odd 2.50-2.99 / Value 1.01-1.50']['stake'], 2)
                elif float(row[8]) < 2.01:
                    odd_value_intervals['Odd 2.50-2.99 / Value 1.51-2.00']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 2.50-2.99 / Value 1.51-2.00']['stake'] += 1
                    odd_value_intervals['Odd 2.50-2.99 / Value 1.51-2.00']['yield'] = round(odd_value_intervals['Odd 2.50-2.99 / Value 1.51-2.00']['units'] * 100 / odd_value_intervals['Odd 2.50-2.99 / Value 1.51-2.00']['stake'], 2)
                elif float(row[8]) < 2.51:
                    odd_value_intervals['Odd 2.50-2.99 / Value 2.01-2.50']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 2.50-2.99 / Value 2.01-2.50']['stake'] += 1
                    odd_value_intervals['Odd 2.50-2.99 / Value 2.01-2.50']['yield'] = round(odd_value_intervals['Odd 2.50-2.99 / Value 2.01-2.50']['units'] * 100 / odd_value_intervals['Odd 2.50-2.99 / Value 2.01-2.50']['stake'], 2)
                elif float(row[8]) < 3.01:
                    odd_value_intervals['Odd 2.50-2.99 / Value 2.51-3.00']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 2.50-2.99 / Value 2.51-3.00']['stake'] += 1
                    odd_value_intervals['Odd 2.50-2.99 / Value 2.51-3.00']['yield'] = round(odd_value_intervals['Odd 2.50-2.99 / Value 2.51-3.00']['units'] * 100 / odd_value_intervals['Odd 2.50-2.99 / Value 2.51-3.00']['stake'], 2)
                else:
                    odd_value_intervals['Odd 2.50-2.99 / Value >3.00']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd 2.50-2.99 / Value >3.00']['stake'] += 1
                    odd_value_intervals['Odd 2.50-2.99 / Value >3.00']['yield'] = round(odd_value_intervals['Odd 2.50-2.99 / Value >3.00']['units'] * 100 / odd_value_intervals['Odd 2.50-2.99 / Value >3.00']['stake'], 2)
            else:
                intervals['>2.99']['units'] += round(units_pick, 2)
                intervals['>2.99']['stake'] += 1
                intervals['>2.99']['yield'] = round(intervals['>2.99']['units'] * 100 / intervals['>2.99']['stake'], 2)

                if float(row[8]) < 1.01:
                    odd_value_intervals['Odd >2.99 / Value <1.01']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd >2.99 / Value <1.01']['stake'] += 1
                    odd_value_intervals['Odd >2.99 / Value <1.01']['yield'] = round(odd_value_intervals['Odd >2.99 / Value <1.01']['units'] * 100 / odd_value_intervals['Odd >2.99 / Value <1.01']['stake'], 2)
                elif float(row[8]) < 1.51:
                    odd_value_intervals['Odd >2.99 / Value 1.01-1.50']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd >2.99 / Value 1.01-1.50']['stake'] += 1
                    odd_value_intervals['Odd >2.99 / Value 1.01-1.50']['yield'] = round(odd_value_intervals['Odd >2.99 / Value 1.01-1.50']['units'] * 100 / odd_value_intervals['Odd >2.99 / Value 1.01-1.50']['stake'], 2)
                elif float(row[8]) < 2.01:
                    odd_value_intervals['Odd >2.99 / Value 1.51-2.00']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd >2.99 / Value 1.51-2.00']['stake'] += 1
                    odd_value_intervals['Odd >2.99 / Value 1.51-2.00']['yield'] = round(odd_value_intervals['Odd >2.99 / Value 1.51-2.00']['units'] * 100 / odd_value_intervals['Odd >2.99 / Value 1.51-2.00']['stake'], 2)
                elif float(row[8]) < 2.51:
                    odd_value_intervals['Odd >2.99 / Value 2.01-2.50']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd >2.99 / Value 2.01-2.50']['stake'] += 1
                    odd_value_intervals['Odd >2.99 / Value 2.01-2.50']['yield'] = round(odd_value_intervals['Odd >2.99 / Value 2.01-2.50']['units'] * 100 / odd_value_intervals['Odd >2.99 / Value 2.01-2.50']['stake'], 2)
                elif float(row[8]) < 3.01:
                    odd_value_intervals['Odd >2.99 / Value 2.51-3.00']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd >2.99 / Value 2.51-3.00']['stake'] += 1
                    odd_value_intervals['Odd >2.99 / Value 2.51-3.00']['yield'] = round(odd_value_intervals['Odd >2.99 / Value 2.51-3.00']['units'] * 100 / odd_value_intervals['Odd >2.99 / Value 2.51-3.00']['stake'], 2)
                else:
                    odd_value_intervals['Odd >2.99 / Value >3.00']['units'] += round(units_pick, 2)
                    odd_value_intervals['Odd >2.99 / Value >3.00']['stake'] += 1
                    odd_value_intervals['Odd >2.99 / Value >3.00']['yield'] = round(odd_value_intervals['Odd >2.99 / Value >3.00']['units'] * 100 / odd_value_intervals['Odd >2.99 / Value >3.00']['stake'], 2)

            # Value intervals
            if float(row[8]) < 1.01:
                value_intervals['<1.01']['units'] += round(units_pick, 2)
                value_intervals['<1.01']['stake'] += 1
                value_intervals['<1.01']['yield'] = round(value_intervals['<1.01']['units'] * 100 / value_intervals['<1.01']['stake'], 2)
            elif float(row[8]) < 1.51:
                value_intervals['1.01-1.50']['units'] += round(units_pick, 2)
                value_intervals['1.01-1.50']['stake'] += 1
                value_intervals['1.01-1.50']['yield'] = round(value_intervals['1.01-1.50']['units'] * 100 / value_intervals['1.01-1.50']['stake'], 2)
            elif float(row[8]) < 2.01:
                value_intervals['1.51-2.00']['units'] += round(units_pick, 2)
                value_intervals['1.51-2.00']['stake'] += 1
                value_intervals['1.51-2.00']['yield'] = round(value_intervals['1.51-2.00']['units'] * 100 / value_intervals['1.51-2.00']['stake'], 2)
            elif float(row[8]) < 2.51:
                value_intervals['2.01-2.50']['units'] += round(units_pick, 2)
                value_intervals['2.01-2.50']['stake'] += 1
                value_intervals['2.01-2.50']['yield'] = round(value_intervals['2.01-2.50']['units'] * 100 / value_intervals['2.01-2.50']['stake'], 2)
            elif float(row[8]) < 3.01:
                value_intervals['2.51-3.00']['units'] += round(units_pick, 2)
                value_intervals['2.51-3.00']['stake'] += 1
                value_intervals['2.51-3.00']['yield'] = round(value_intervals['2.51-3.00']['units'] * 100 / value_intervals['2.51-3.00']['stake'], 2)
            else:
                value_intervals['>3.00']['units'] += round(units_pick, 2)
                value_intervals['>3.00']['stake'] += 1
                value_intervals['>3.00']['yield'] = round(value_intervals['>3.00']['units'] * 100 / value_intervals['>3.00']['stake'], 2)
        else:
            row.append(Color('{autored}L{/autored}'))
            row.append(-1)
            units -= 1

            # Odds intervals
            if float(units_pick) < 0.50:
                intervals['<1.50']['units'] -= 1
                intervals['<1.50']['stake'] += 1
                intervals['<1.50']['yield'] = round(intervals['<1.50']['units'] * 100 / intervals['<1.50']['stake'], 2)

                if float(row[8]) < 1.01:
                    odd_value_intervals['Odd <1.50 / Value <1.01']['units'] -= 1
                    odd_value_intervals['Odd <1.50 / Value <1.01']['stake'] += 1
                    odd_value_intervals['Odd <1.50 / Value <1.01']['yield'] = round(odd_value_intervals['Odd <1.50 / Value <1.01']['units'] * 100 / odd_value_intervals['Odd <1.50 / Value <1.01']['stake'], 2)
                elif float(row[8]) < 1.51:
                    odd_value_intervals['Odd <1.50 / Value 1.01-1.50']['units'] -= 1
                    odd_value_intervals['Odd <1.50 / Value 1.01-1.50']['stake'] += 1
                    odd_value_intervals['Odd <1.50 / Value 1.01-1.50']['yield'] = round(odd_value_intervals['Odd <1.50 / Value 1.01-1.50']['units'] * 100 / odd_value_intervals['Odd <1.50 / Value 1.01-1.50']['stake'], 2)
                elif float(row[8]) < 2.01:
                    odd_value_intervals['Odd <1.50 / Value 1.51-2.00']['units'] -= 1
                    odd_value_intervals['Odd <1.50 / Value 1.51-2.00']['stake'] += 1
                    odd_value_intervals['Odd <1.50 / Value 1.51-2.00']['yield'] = round(odd_value_intervals['Odd <1.50 / Value 1.51-2.00']['units'] * 100 / odd_value_intervals['Odd <1.50 / Value 1.51-2.00']['stake'], 2)
                elif float(row[8]) < 2.51:
                    odd_value_intervals['Odd <1.50 / Value 2.01-2.50']['units'] -= 1
                    odd_value_intervals['Odd <1.50 / Value 2.01-2.50']['stake'] += 1
                    odd_value_intervals['Odd <1.50 / Value 2.01-2.50']['yield'] = round(odd_value_intervals['Odd <1.50 / Value 2.01-2.50']['units'] * 100 / odd_value_intervals['Odd <1.50 / Value 2.01-2.50']['stake'], 2)
                elif float(row[8]) < 3.01:
                    odd_value_intervals['Odd <1.50 / Value 2.51-3.00']['units'] -= 1
                    odd_value_intervals['Odd <1.50 / Value 2.51-3.00']['stake'] += 1
                    odd_value_intervals['Odd <1.50 / Value 2.51-3.00']['yield'] = round(odd_value_intervals['Odd <1.50 / Value 2.51-3.00']['units'] * 100 / odd_value_intervals['Odd <1.50 / Value 2.51-3.00']['stake'], 2)
                else:
                    odd_value_intervals['Odd <1.50 / Value >3.00']['units'] -= 1
                    odd_value_intervals['Odd <1.50 / Value >3.00']['stake'] += 1
                    odd_value_intervals['Odd <1.50 / Value >3.00']['yield'] = round(odd_value_intervals['Odd <1.50 / Value >3.00']['units'] * 100 / odd_value_intervals['Odd <1.50 / Value >3.00']['stake'], 2)
            elif float(units_pick) < 1.00:
                intervals['1.50-1.99']['units'] -= 1
                intervals['1.50-1.99']['stake'] += 1
                intervals['1.50-1.99']['yield'] = round(intervals['1.50-1.99']['units'] * 100 / intervals['1.50-1.99']['stake'], 2)

                if float(row[8]) < 1.01:
                    odd_value_intervals['Odd 1.50-1.99 / Value <1.01']['units'] -= 1
                    odd_value_intervals['Odd 1.50-1.99 / Value <1.01']['stake'] += 1
                    odd_value_intervals['Odd 1.50-1.99 / Value <1.01']['yield'] = round(odd_value_intervals['Odd 1.50-1.99 / Value <1.01']['units'] * 100 / odd_value_intervals['Odd 1.50-1.99 / Value <1.01']['stake'], 2)
                elif float(row[8]) < 1.51:
                    odd_value_intervals['Odd 1.50-1.99 / Value 1.01-1.50']['units'] -= 1
                    odd_value_intervals['Odd 1.50-1.99 / Value 1.01-1.50']['stake'] += 1
                    odd_value_intervals['Odd 1.50-1.99 / Value 1.01-1.50']['yield'] = round(odd_value_intervals['Odd 1.50-1.99 / Value 1.01-1.50']['units'] * 100 / odd_value_intervals['Odd 1.50-1.99 / Value 1.01-1.50']['stake'], 2)
                elif float(row[8]) < 2.01:
                    odd_value_intervals['Odd 1.50-1.99 / Value 1.51-2.00']['units'] -= 1
                    odd_value_intervals['Odd 1.50-1.99 / Value 1.51-2.00']['stake'] += 1
                    odd_value_intervals['Odd 1.50-1.99 / Value 1.51-2.00']['yield'] = round(odd_value_intervals['Odd 1.50-1.99 / Value 1.51-2.00']['units'] * 100 / odd_value_intervals['Odd 1.50-1.99 / Value 1.51-2.00']['stake'], 2)
                elif float(row[8]) < 2.51:
                    odd_value_intervals['Odd 1.50-1.99 / Value 2.01-2.50']['units'] -= 1
                    odd_value_intervals['Odd 1.50-1.99 / Value 2.01-2.50']['stake'] += 1
                    odd_value_intervals['Odd 1.50-1.99 / Value 2.01-2.50']['yield'] = round(odd_value_intervals['Odd 1.50-1.99 / Value 2.01-2.50']['units'] * 100 / odd_value_intervals['Odd 1.50-1.99 / Value 2.01-2.50']['stake'], 2)
                elif float(row[8]) < 3.01:
                    odd_value_intervals['Odd 1.50-1.99 / Value 2.51-3.00']['units'] -= 1
                    odd_value_intervals['Odd 1.50-1.99 / Value 2.51-3.00']['stake'] += 1
                    odd_value_intervals['Odd 1.50-1.99 / Value 2.51-3.00']['yield'] = round(odd_value_intervals['Odd 1.50-1.99 / Value 2.51-3.00']['units'] * 100 / odd_value_intervals['Odd 1.50-1.99 / Value 2.51-3.00']['stake'], 2)
                else:
                    odd_value_intervals['Odd 1.50-1.99 / Value >3.00']['units'] -= 1
                    odd_value_intervals['Odd 1.50-1.99 / Value >3.00']['stake'] += 1
                    odd_value_intervals['Odd 1.50-1.99 / Value >3.00']['yield'] = round(odd_value_intervals['Odd 1.50-1.99 / Value >3.00']['units'] * 100 / odd_value_intervals['Odd 1.50-1.99 / Value >3.00']['stake'], 2)
            elif float(units_pick) < 1.50:
                intervals['2.00-2.49']['units'] -= 1
                intervals['2.00-2.49']['stake'] += 1
                intervals['2.00-2.49']['yield'] = round(intervals['2.00-2.49']['units'] * 100 / intervals['2.00-2.49']['stake'], 2)

                if float(row[8]) < 1.01:
                    odd_value_intervals['Odd 2.00-2.49 / Value <1.01']['units'] -= 1
                    odd_value_intervals['Odd 2.00-2.49 / Value <1.01']['stake'] += 1
                    odd_value_intervals['Odd 2.00-2.49 / Value <1.01']['yield'] = round(odd_value_intervals['Odd 2.00-2.49 / Value <1.01']['units'] * 100 / odd_value_intervals['Odd 2.00-2.49 / Value <1.01']['stake'], 2)
                elif float(row[8]) < 1.51:
                    odd_value_intervals['Odd 2.00-2.49 / Value 1.01-1.50']['units'] -= 1
                    odd_value_intervals['Odd 2.00-2.49 / Value 1.01-1.50']['stake'] += 1
                    odd_value_intervals['Odd 2.00-2.49 / Value 1.01-1.50']['yield'] = round(odd_value_intervals['Odd 2.00-2.49 / Value 1.01-1.50']['units'] * 100 / odd_value_intervals['Odd 2.00-2.49 / Value 1.01-1.50']['stake'], 2)
                elif float(row[8]) < 2.01:
                    odd_value_intervals['Odd 2.00-2.49 / Value 1.51-2.00']['units'] -= 1
                    odd_value_intervals['Odd 2.00-2.49 / Value 1.51-2.00']['stake'] += 1
                    odd_value_intervals['Odd 2.00-2.49 / Value 1.51-2.00']['yield'] = round(odd_value_intervals['Odd 2.00-2.49 / Value 1.51-2.00']['units'] * 100 / odd_value_intervals['Odd 2.00-2.49 / Value 1.51-2.00']['stake'], 2)
                elif float(row[8]) < 2.51:
                    odd_value_intervals['Odd 2.00-2.49 / Value 2.01-2.50']['units'] -= 1
                    odd_value_intervals['Odd 2.00-2.49 / Value 2.01-2.50']['stake'] += 1
                    odd_value_intervals['Odd 2.00-2.49 / Value 2.01-2.50']['yield'] = round(odd_value_intervals['Odd 2.00-2.49 / Value 2.01-2.50']['units'] * 100 / odd_value_intervals['Odd 2.00-2.49 / Value 2.01-2.50']['stake'], 2)
                elif float(row[8]) < 3.01:
                    odd_value_intervals['Odd 2.00-2.49 / Value 2.51-3.00']['units'] -= 1
                    odd_value_intervals['Odd 2.00-2.49 / Value 2.51-3.00']['stake'] += 1
                    odd_value_intervals['Odd 2.00-2.49 / Value 2.51-3.00']['yield'] = round(odd_value_intervals['Odd 2.00-2.49 / Value 2.51-3.00']['units'] * 100 / odd_value_intervals['Odd 2.00-2.49 / Value 2.51-3.00']['stake'], 2)
                else:
                    odd_value_intervals['Odd 2.00-2.49 / Value >3.00']['units'] -= 1
                    odd_value_intervals['Odd 2.00-2.49 / Value >3.00']['stake'] += 1
                    odd_value_intervals['Odd 2.00-2.49 / Value >3.00']['yield'] = round(odd_value_intervals['Odd 2.00-2.49 / Value >3.00']['units'] * 100 / odd_value_intervals['Odd 2.00-2.49 / Value >3.00']['stake'], 2)
            elif float(units_pick) < 2.00:
                intervals['2.50-2.99']['units'] -= 1
                intervals['2.50-2.99']['stake'] += 1
                intervals['2.50-2.99']['yield'] = round(intervals['2.50-2.99']['units'] * 100 / intervals['2.50-2.99']['stake'], 2)

                if float(row[8]) < 1.01:
                    odd_value_intervals['Odd 2.50-2.99 / Value <1.01']['units'] -= 1
                    odd_value_intervals['Odd 2.50-2.99 / Value <1.01']['stake'] += 1
                    odd_value_intervals['Odd 2.50-2.99 / Value <1.01']['yield'] = round(odd_value_intervals['Odd 2.50-2.99 / Value <1.01']['units'] * 100 / odd_value_intervals['Odd 2.50-2.99 / Value <1.01']['stake'], 2)
                elif float(row[8]) < 1.51:
                    odd_value_intervals['Odd 2.50-2.99 / Value 1.01-1.50']['units'] -= 1
                    odd_value_intervals['Odd 2.50-2.99 / Value 1.01-1.50']['stake'] += 1
                    odd_value_intervals['Odd 2.50-2.99 / Value 1.01-1.50']['yield'] = round(odd_value_intervals['Odd 2.50-2.99 / Value 1.01-1.50']['units'] * 100 / odd_value_intervals['Odd 2.50-2.99 / Value 1.01-1.50']['stake'], 2)
                elif float(row[8]) < 2.01:
                    odd_value_intervals['Odd 2.50-2.99 / Value 1.51-2.00']['units'] -= 1
                    odd_value_intervals['Odd 2.50-2.99 / Value 1.51-2.00']['stake'] += 1
                    odd_value_intervals['Odd 2.50-2.99 / Value 1.51-2.00']['yield'] = round(odd_value_intervals['Odd 2.50-2.99 / Value 1.51-2.00']['units'] * 100 / odd_value_intervals['Odd 2.50-2.99 / Value 1.51-2.00']['stake'], 2)
                elif float(row[8]) < 2.51:
                    odd_value_intervals['Odd 2.50-2.99 / Value 2.01-2.50']['units'] -= 1
                    odd_value_intervals['Odd 2.50-2.99 / Value 2.01-2.50']['stake'] += 1
                    odd_value_intervals['Odd 2.50-2.99 / Value 2.01-2.50']['yield'] = round(odd_value_intervals['Odd 2.50-2.99 / Value 2.01-2.50']['units'] * 100 / odd_value_intervals['Odd 2.50-2.99 / Value 2.01-2.50']['stake'], 2)
                elif float(row[8]) < 3.01:
                    odd_value_intervals['Odd 2.50-2.99 / Value 2.51-3.00']['units'] -= 1
                    odd_value_intervals['Odd 2.50-2.99 / Value 2.51-3.00']['stake'] += 1
                    odd_value_intervals['Odd 2.50-2.99 / Value 2.51-3.00']['yield'] = round(odd_value_intervals['Odd 2.50-2.99 / Value 2.51-3.00']['units'] * 100 / odd_value_intervals['Odd 2.50-2.99 / Value 2.51-3.00']['stake'], 2)
                else:
                    odd_value_intervals['Odd 2.50-2.99 / Value >3.00']['units'] -= 1
                    odd_value_intervals['Odd 2.50-2.99 / Value >3.00']['stake'] += 1
                    odd_value_intervals['Odd 2.50-2.99 / Value >3.00']['yield'] = round(odd_value_intervals['Odd 2.50-2.99 / Value >3.00']['units'] * 100 / odd_value_intervals['Odd 2.50-2.99 / Value >3.00']['stake'], 2)
            else:
                intervals['>2.99']['units'] -= 1
                intervals['>2.99']['stake'] += 1
                intervals['>2.99']['yield'] = round(intervals['>2.99']['units'] * 100 / intervals['>2.99']['stake'], 2)

                if float(row[8]) < 1.01:
                    odd_value_intervals['Odd >2.99 / Value <1.01']['units'] -= 1
                    odd_value_intervals['Odd >2.99 / Value <1.01']['stake'] += 1
                    odd_value_intervals['Odd >2.99 / Value <1.01']['yield'] = round(odd_value_intervals['Odd >2.99 / Value <1.01']['units'] * 100 / odd_value_intervals['Odd >2.99 / Value <1.01']['stake'], 2)
                elif float(row[8]) < 1.51:
                    odd_value_intervals['Odd >2.99 / Value 1.01-1.50']['units'] -= 1
                    odd_value_intervals['Odd >2.99 / Value 1.01-1.50']['stake'] += 1
                    odd_value_intervals['Odd >2.99 / Value 1.01-1.50']['yield'] = round(odd_value_intervals['Odd >2.99 / Value 1.01-1.50']['units'] * 100 / odd_value_intervals['Odd >2.99 / Value 1.01-1.50']['stake'], 2)
                elif float(row[8]) < 2.01:
                    odd_value_intervals['Odd >2.99 / Value 1.51-2.00']['units'] -= 1
                    odd_value_intervals['Odd >2.99 / Value 1.51-2.00']['stake'] += 1
                    odd_value_intervals['Odd >2.99 / Value 1.51-2.00']['yield'] = round(odd_value_intervals['Odd >2.99 / Value 1.51-2.00']['units'] * 100 / odd_value_intervals['Odd >2.99 / Value 1.51-2.00']['stake'], 2)
                elif float(row[8]) < 2.51:
                    odd_value_intervals['Odd >2.99 / Value 2.01-2.50']['units'] -= 1
                    odd_value_intervals['Odd >2.99 / Value 2.01-2.50']['stake'] += 1
                    odd_value_intervals['Odd >2.99 / Value 2.01-2.50']['yield'] = round(odd_value_intervals['Odd >2.99 / Value 2.01-2.50']['units'] * 100 / odd_value_intervals['Odd >2.99 / Value 2.01-2.50']['stake'], 2)
                elif float(row[8]) < 3.01:
                    odd_value_intervals['Odd >2.99 / Value 2.51-3.00']['units'] -= 1
                    odd_value_intervals['Odd >2.99 / Value 2.51-3.00']['stake'] += 1
                    odd_value_intervals['Odd >2.99 / Value 2.51-3.00']['yield'] = round(odd_value_intervals['Odd >2.99 / Value 2.51-3.00']['units'] * 100 / odd_value_intervals['Odd >2.99 / Value 2.51-3.00']['stake'], 2)
                else:
                    odd_value_intervals['Odd >2.99 / Value >3.00']['units'] -= 1
                    odd_value_intervals['Odd >2.99 / Value >3.00']['stake'] += 1
                    odd_value_intervals['Odd >2.99 / Value >3.00']['yield'] = round(odd_value_intervals['Odd >2.99 / Value >3.00']['units'] * 100 / odd_value_intervals['Odd >2.99 / Value >3.00']['stake'], 2)

            # Value intervals
            if float(row[8]) < 1.01:
                value_intervals['<1.01']['units'] -= 1
                value_intervals['<1.01']['stake'] += 1
                value_intervals['<1.01']['yield'] = round(value_intervals['<1.01']['units'] * 100 / value_intervals['<1.01']['stake'], 2)
            elif float(row[8]) < 1.51:
                value_intervals['1.01-1.50']['units'] -= 1
                value_intervals['1.01-1.50']['stake'] += 1
                value_intervals['1.01-1.50']['yield'] = round(value_intervals['1.01-1.50']['units'] * 100 / value_intervals['1.01-1.50']['stake'], 2)
            elif float(row[8]) < 2.01:
                value_intervals['1.51-2.00']['units'] -= 1
                value_intervals['1.51-2.00']['stake'] += 1
                value_intervals['1.51-2.00']['yield'] = round(value_intervals['1.51-2.00']['units'] * 100 / value_intervals['1.51-2.00']['stake'], 2)
            elif float(row[8]) < 2.51:
                value_intervals['2.01-2.50']['units'] -= 1
                value_intervals['2.01-2.50']['stake'] += 1
                value_intervals['2.01-2.50']['yield'] = round(value_intervals['2.01-2.50']['units'] * 100 / value_intervals['2.01-2.50']['stake'], 2)
            elif float(row[8]) < 3.01:
                value_intervals['2.51-3.00']['units'] -= 1
                value_intervals['2.51-3.00']['stake'] += 1
                value_intervals['2.51-3.00']['yield'] = round(value_intervals['2.51-3.00']['units'] * 100 / value_intervals['2.51-3.00']['stake'], 2)
            else:
                value_intervals['>3.00']['units'] -= 1
                value_intervals['>3.00']['stake'] += 1
                value_intervals['>3.00']['yield'] = round(value_intervals['>3.00']['units'] * 100 / value_intervals['>3.00']['stake'], 2)

        row.append(coef[0])
        prediction_table.append(row)
    index += 1

table_instance = SingleTable(prediction_table, Color('{autocyan} Prediction with new games {/autocyan}'))
table_instance.inner_heading_row_border = False
table_instance.inner_row_border = True
table_instance.justify_columns = {0: 'left', 1: 'center', 2: 'center', 3: 'center', 4: 'center', 5: 'center', 6: 'left', 7: 'center', 8: 'center', 9: 'center', 10: 'center', 11: 'center'}
print("\n" + table_instance.table)

print("\n" + Color('{autogreen}Units: {/autogreen}') + str(round(units, 2)))
print(Color('{autogreen}Hits: {/autogreen}') + str(hits))
print(Color('{autogreen}Stake: {/autogreen}') + str(stake))
print(Color('{autogreen}Yield: {/autogreen}') + str(round(units * 100 / stake, 2)) + "%")
print(Color('{autogreen}Different probabilities: {/autogreen}') + str(len(different_probs)))
print(Color('{autogreen}Value intervals: {/autogreen}'))
pprint(value_intervals)
print(Color('{autogreen}Odd intervals: {/autogreen}'))
pprint(intervals)
print(Color('{autogreen}Odd/Value intervals: {/autogreen}'))
pprint(odd_value_intervals)

# Gràfica
plt.figure("The Beast Training")
plt.title("Test 22/04/2019")

index = 0
axes = []

while index < num_epochs:
    axes.append(index)
    index += (num_epochs / 200)

plt.plot(axes, values)
plt.plot(axes, values_predicted, color='r')
plt.ylim([0, 100])
plt.xlim([0, num_epochs])
plt.ylabel('Error')
plt.xlabel('Training Time')
plt.tight_layout()
plt.show()

# Mostrar pesos
#nn.print_weights()
