from cassandra.cluster import Cluster
import numpy as np
import matplotlib.pyplot as plt
from colorclass import Color, Windows
from terminaltables import SingleTable

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
            if k % (epochs / 100) == 0:
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
surfaces = ["H", "C", "G", "I"]
games = []
winners_train = []
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
num_epochs = 500000
nn = NeuralNetwork([46, 10, 1], activation = 'tanh')

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
num_games_train = round(num_games * 0.8, 1)

query = "SELECT game_season, game_tournament, game_week, game_round, game_surface, game_category, game_sets, game_points, game_date, game_rank1, game_rank2, game_race1, game_race2, game_rankmax1, game_rankmax2, game_age1, game_age2, game_h2h1, game_h2h2, game_h2h_year1, game_h2h_year2, game_h2h_surface1, game_h2h_surface2, game_h2h_surface_year1, game_h2h_surface_year2, game_surface1, game_surface2, game_surface_year1, game_surface_year2, game_hand1, game_hand2, game_home1, game_home2, game_3months1, game_3months2, game_10streak1, game_10streak2, game_gp1m1, game_gp1m2, game_gp3m1, game_gp3m2, game_gp6m1, game_gp6m2, game_pts_def1, game_pts_def2, game_player1, game_player2, game_winner, game_odd1, game_odd2 FROM game_train"
games_db = session.execute(query)
index = 0

for game_db in games_db:
    game = []
    game.append(game_db.game_season)
    game.append(game_db.game_week)
    game.append(rounds.index(game_db.game_round))
    game.append(surfaces.index(game_db.game_surface))
    game.append(int(game_db.game_category))
    game.append(game_db.game_sets)
    game.append(int(str(game_db.game_date)[5:7]))
    game.append(game_db.game_rank1)
    game.append(game_db.game_rank2)
    game.append(game_db.game_race1)
    game.append(game_db.game_race2)
    game.append(game_db.game_rankmax1)
    game.append(game_db.game_rankmax2)
    game.append(game_db.game_age1)
    game.append(game_db.game_age2)
    game.append(game_db.game_h2h1)
    game.append(game_db.game_h2h2)
    game.append(game_db.game_h2h_year1)
    game.append(game_db.game_h2h_year2)
    game.append(game_db.game_h2h_surface1)
    game.append(game_db.game_h2h_surface2)
    game.append(game_db.game_h2h_surface_year1)
    game.append(game_db.game_h2h_surface_year2)
    game.append(game_db.game_surface1)
    game.append(game_db.game_surface2)
    game.append(game_db.game_surface_year1)
    game.append(game_db.game_surface_year2)
    game.append(game_db.game_hand1)
    game.append(game_db.game_hand2)
    game.append(game_db.game_home1)
    game.append(game_db.game_home2)
    game.append(game_db.game_3months1)
    game.append(game_db.game_3months2)
    game.append(game_db.game_10streak1)
    game.append(game_db.game_10streak2)
    game.append(game_db.game_gp1m1)
    game.append(game_db.game_gp1m2)
    game.append(game_db.game_gp3m1)
    game.append(game_db.game_gp3m2)
    game.append(game_db.game_gp6m1)
    game.append(game_db.game_gp6m2)
    game.append(game_db.game_pts_def1)
    game.append(game_db.game_pts_def2)
    game.append(game_db.game_points)
    game.append(game_db.game_odd1)
    game.append(game_db.game_odd2)

    query_player1 = "SELECT player_name FROM player_by_atpid WHERE player_atpwt_id = '" + game_db.game_player1 + "'"
    player1 = session.execute(query_player1)
    player1_name = player1[0].player_name
    query_player2 = "SELECT player_name FROM player_by_atpid WHERE player_atpwt_id = '" + game_db.game_player2 + "'"
    player2 = session.execute(query_player2)
    player2_name = player2[0].player_name

    if index < num_games_train:
        games.append(game)
        games_names_train.append(player1_name + " - " + player2_name)
        dates_train.append(str(game_db.game_date))
        tournaments_train.append(game_db.game_tournament.capitalize())

        if game_db.game_winner == game_db.game_player1:
            if game_db.game_odd1 <= game_db.game_odd2:
                winners_train.append([-1])
            else:
                winners_train.append([1])
        elif game_db.game_winner == game_db.game_player2:
            if game_db.game_odd2 <= game_db.game_odd1:
                winners_train.append([-1])
            else:
                winners_train.append([1])
    else:
        predict.append(game)
        games_names_predict.append(player1_name + " - " + player2_name)
        dates_predict.append(str(game_db.game_date))
        tournaments_predict.append(game_db.game_tournament.capitalize())

        if game_db.game_winner == game_db.game_player1:
            if game_db.game_odd1 <= game_db.game_odd2:
                winners_sim.append([-1])
            else:
                winners_sim.append([1])
        elif game_db.game_winner == game_db.game_player2:
            if game_db.game_odd2 <= game_db.game_odd1:
                winners_sim.append([-1])
            else:
                winners_sim.append([1])

    index += 1

np.set_printoptions(suppress=True)
X = np.array(games)
y = np.array(winners_train)
Z = np.array(predict)
sol = np.array(winners_sim)

# Previous prediction
index = 0

for e in X:
    print("X:", games_names_train[index], "Sol:", y[index], "Network:", nn.predict(e))
    index += 1

nn.fit(X, y, learning_rate = 0.03, epochs = num_epochs)

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

for e in X:
    stake += 1
    row = []
    row.append(games_names_train[index])
    players = games_names_train[index].split(" - ")
    row.append(dates_train[index])
    row.append(tournaments_train[index])
    row.append(rounds[games[index][2]])
    row.append(games[index][44])
    row.append(games[index][45])

    coef = nn.predict(e)

    if coef <= 0:
        if games[index][44] <= games[index][45]:
            row.append(players[0])
        else:
            row.append(players[1])
    else:
        if games[index][44] <= games[index][45]:
            row.append(players[1])
        else:
            row.append(players[0])

    prob_opp = round((coef[0] + 1) * 100 / 2, 2)
    prob_fav = round(100 - prob_opp, 2)

    if prob_opp > prob_fav:
        row.append(str(prob_opp) + "%")
    else:
        row.append(str(prob_fav) + "%")

    inv_odd1 = 1 / games[index][44]
    inv_odd2 = 1 / games[index][45]
    inv_tot = inv_odd1 + inv_odd2
    prob_bookmark1 = round(inv_odd1 * (100 - (inv_tot * 100 - 100)), 2)
    prob_bookmark2 = round(inv_odd2 * (100 - (inv_tot * 100 - 100)), 2)
    new_odd1 = round(100 / prob_bookmark1, 2)
    new_odd2 = round(100 / prob_bookmark2, 2)

    if games[index][44] <= games[index][45]:
        value1 = round(games[index][44] * prob_fav / 100, 2)
        value2 = round(games[index][45] * prob_opp / 100, 2)
        #value1 = round((new_odd1 * prob_fav / 100 - 1) * 100, 2)
        #value2 = round((new_odd2 * prob_opp / 100 - 1) * 100, 2)
    else:
        value1 = round(games[index][45] * prob_fav / 100, 2)
        value2 = round(games[index][44] * prob_opp / 100, 2)

    if prob_opp > prob_fav:
        row.append(str(value2))
    else:
        row.append(str(value1))

    if coef <= 0:
        if y[index] == -1:
            row.append(Color('{autogreen}W{/autogreen}'))

            if games[index][44] <= games[index][45]:
                row.append("+" + str(round(games[index][44] - 1, 2)))
                units += round(games[index][44] - 1, 2)
            else:
                row.append("+" + str(round(games[index][45] - 1, 2)))
                units += round(games[index][45] - 1, 2)
        else:
            row.append(Color('{autored}L{/autored}'))
            row.append(-1)
            units -= 1
    else:
        if y[index] == 1:
            row.append(Color('{autogreen}W{/autogreen}'))

            if games[index][44] > games[index][45]:
                row.append("+" + str(round(games[index][44] - 1, 2)))
                units += round(games[index][44] - 1, 2)
            else:
                row.append("+" + str(round(games[index][45] - 1, 2)))
                units += round(games[index][45] - 1, 2)
        else:
            row.append(Color('{autored}L{/autored}'))
            row.append(-1)
            units -= 1

    row.append(coef)
    train_table.append(row)
    index += 1

table_instance = SingleTable(train_table, Color('{autocyan} Prediction with training games {/autocyan}'))
table_instance.inner_heading_row_border = False
table_instance.inner_row_border = True
table_instance.justify_columns = {0: 'left', 1: 'center', 2: 'center', 3: 'center', 4: 'center', 5: 'center', 6: 'left', 7: 'center', 8: 'center', 9: 'center', 10: 'center', 11: 'center'}
print("\n" + table_instance.table)

print("\n" + Color('{autogreen}Units: {/autogreen}') + str(round(units, 2)))
print(Color('{autogreen}Stake: {/autogreen}') + str(stake))
print(Color('{autogreen}Yield: {/autogreen}') + str(round(units * 100 / stake, 2)) + "%")

# Prediction with new games

index = 0
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

for e in Z:
    stake += 1
    row = []
    row.append(games_names_predict[index])
    players = games_names_predict[index].split(" - ")
    row.append(dates_predict[index])
    row.append(tournaments_predict[index])
    row.append(rounds[games[index][2]])
    row.append(games[index][44])
    row.append(games[index][45])

    coef = nn.predict(e)

    if coef <= 0:
        if games[index][44] <= games[index][45]:
            row.append(players[0])
        else:
            row.append(players[1])
    else:
        if games[index][44] <= games[index][45]:
            row.append(players[1])
        else:
            row.append(players[0])

    prob_opp = round((coef[0] + 1) * 100 / 2, 2)
    prob_fav = round(100 - prob_opp, 2)

    if prob_opp > prob_fav:
        row.append(str(prob_opp) + "%")
    else:
        row.append(str(prob_fav) + "%")

    inv_odd1 = 1 / games[index][44]
    inv_odd2 = 1 / games[index][45]
    inv_tot = inv_odd1 + inv_odd2
    prob_bookmark1 = round(inv_odd1 * (100 - (inv_tot * 100 - 100)), 2)
    prob_bookmark2 = round(inv_odd2 * (100 - (inv_tot * 100 - 100)), 2)
    new_odd1 = round(100 / prob_bookmark1, 2)
    new_odd2 = round(100 / prob_bookmark2, 2)

    if games[index][44] <= games[index][45]:
        value1 = round(games[index][44] * prob_fav / 100, 2)
        value2 = round(games[index][45] * prob_opp / 100, 2)
        #value1 = round((new_odd1 * prob_fav / 100 - 1) * 100, 2)
        #value2 = round((new_odd2 * prob_opp / 100 - 1) * 100, 2)
    else:
        value1 = round(games[index][45] * prob_fav / 100, 2)
        value2 = round(games[index][44] * prob_opp / 100, 2)

    if prob_opp > prob_fav:
        row.append(str(value2))
    else:
        row.append(str(value1))

    if coef <= 0:
        if y[index] == -1:
            row.append(Color('{autogreen}W{/autogreen}'))

            if games[index][44] <= games[index][45]:
                row.append("+" + str(round(games[index][44] - 1, 2)))
                units += round(games[index][44] - 1, 2)
            else:
                row.append("+" + str(round(games[index][45] - 1, 2)))
                units += round(games[index][45] - 1, 2)
        else:
            row.append(Color('{autored}L{/autored}'))
            row.append(-1)
            units -= 1
    else:
        if y[index] == 1:
            row.append(Color('{autogreen}W{/autogreen}'))

            if games[index][44] > games[index][45]:
                row.append("+" + str(round(games[index][44] - 1, 2)))
                units += round(games[index][44] - 1, 2)
            else:
                row.append("+" + str(round(games[index][45] - 1, 2)))
                units += round(games[index][45] - 1, 2)
        else:
            row.append(Color('{autored}L{/autored}'))
            row.append(-1)
            units -= 1

    row.append(coef)
    prediction_table.append(row)
    index += 1

table_instance = SingleTable(prediction_table, Color('{autocyan} Prediction with new games {/autocyan}'))
table_instance.inner_heading_row_border = False
table_instance.inner_row_border = True
table_instance.justify_columns = {0: 'left', 1: 'center', 2: 'center', 3: 'center', 4: 'center', 5: 'center', 6: 'left', 7: 'center', 8: 'center', 9: 'center', 10: 'center', 11: 'center'}
print("\n" + table_instance.table)

print("\n" + Color('{autogreen}Units: {/autogreen}') + str(round(units, 2)))
print(Color('{autogreen}Stake: {/autogreen}') + str(stake))
print(Color('{autogreen}Yield: {/autogreen}') + str(round(units * 100 / stake, 2)) + "%")

# Gràfica
plt.figure("The Beast Training")
plt.title("Test 17/04/2019")

index = 0
axes = []

while index < num_epochs:
    axes.append(index)
    index += (num_epochs / 100)

plt.plot(axes, values)
plt.plot(axes, values_predicted, color='r')
plt.ylim([0, 100])
plt.xlim([0, num_epochs])
plt.ylabel('Error')
plt.xlabel('Training Time')
plt.tight_layout()
plt.show()

# Mostrar pesos
# nn.print_weights()
