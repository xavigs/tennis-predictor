import numpy as np
import matplotlib.pyplot as plt

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

                    if coef[0] < 0.5 and y[i][0] == 1 or coef[0] > 0.5 and y[i][0] == 0:
                        total_errors += 1

                pct_errors = round((total_errors * 100) / total_items, 2)
                values.append(pct_errors)

                # Predicció
                global Z
                global values_predicted
                total_errors = 0
                total_items = len(Z)

                for i, item in enumerate(Z):
                    coef = self.predict(item)

                    if coef[0] < 0.5 and y[i][0] == 1 or coef[0] > 0.5 and y[i][0] == 0:
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

# Test app
values = []
values_predicted = []
num_epochs = 1000000
nn = NeuralNetwork([4, 3, 2], activation = 'tanh')
games = []
predict = []
games_names = ["Isner vs Aliassime"]

'''
    Torneig - Ronda - Local - Edat - Rank - Race - RankMax - H2H - %Any - %AnySup - %SupCar - 3Mesos - PtsDef - Odd
'''

# Basilashvili vs Eubanks (1R)
bas_eub = [2000, 1, 0, 0, 26, 22, 20, 170, 73, 63, 20, 166, 0, 0, 67, 50, 67, 50, 56, 58, 13, 10, 90, 0, 123, 416]
games.append(bas_eub)

# Travaglia vs Andreozzi (1R)
tra_and = [2000, 1, 0, 0, 27, 27, 137, 77, 229, 103, 108, 77, 33, 67, 0, 50, 0, 50, 58, 58, 5, 16, 0, 0, 145, 276]
games.append(tra_and)

# Anderson vs Mannarino (1R)
and_man = [2000, 1, 0, 0, 32, 30, 6, 42, 3, 0, 5, 22, 75, 25, 100, 0, 100, 0, 68, 61, 25, 14, 10, 90, 106, 929]
games.append(and_man)

# Mmoh vs Albot (1R)
mmo_alb = [2000, 1, 0, 0, 21, 29, 107, 98, 101, 120, 96, 81, 0, 0, 50, 0, 50, 0, 64, 66, -3, 0, 0, 10, 171, 213]
games.append(mmo_alb)

# Verdasco vs Kecmanovic (1R)
ver_kec = [2000, 1, 0, 0, 35, 29, 28, 125, 139, 108, 7, 125, 0, 0, 75, 75, 75, 75, 57, 65, 3, 22, 45, 0, 147, 270]
games.append(ver_kec)

# Garcia-López vs Haase (1R)
gar_haa = [2000, 1, 0, 0, 35, 31, 100, 58, 41, 0, 23, 33, 100, 0, 75, 0, 75, 0, 53, 53, -6, -17, 45, 10, 176, 206]
games.append(gar_haa)

# Tsitsipas vs Berrettini (1R)
tsi_ber = [2000, 1, 0, 0, 20, 22, 15, 54, 73, 147, 15, 52, 100, 0, 50, 33, 50, 33, 61, 67, 6, -4, 10, 10, 139, 301]
games.append(tsi_ber)

# Tiafoe vs Gunneswaran (1R)
tia_gun = [2000, 1, 0, 0, 20, 29, 39, 109, 0, 10, 38, 104, 0, 0, 0, 0, 0, 0, 59, 67, 13, 36, 10, 0, 129, 358]
games.append(tia_gun)

# Dimitrov vs Tipsarevic (1R)
dim_tip = [2000, 1, 0, 0, 27, 34, 21, 0, 73, 0, 3, 8, 17, 83, 67, 0, 67, 0, 63, 61, -57, 0, 360, 0, 105, 986]
games.append(dim_tip)

# Nadal vs Duckworth (1R)
nad_duc = [2000, 1, 0, 1, 32, 26, 2, 237, 0, 0, 1, 82, 0, 0, 0, 0, 0, 0, 78, 60, -50, 8, 360, 0, 104, 1108]
games.append(nad_duc)

# Lajovic vs Cuevas (1R)
laj_cue = [2000, 1, 0, 0, 28, 33, 46, 94, 83, 266, 45, 19, 0, 100, 50, 33, 50, 33, 57, 46, 8, -30, 10, 45, 137, 307]
games.append(laj_cue)

# Kubler vs Fabbiano (1R)
kub_fab = [2000, 1, 1, 0, 25, 29, 130, 102, 0, 264, 91, 70, 0, 0, 0, 50, 0, 50, 65, 65, -30, 22, 10, 10, 181, 200]
games.append(kub_fab)

# Isner vs Opelka (1R)
isn_ope = [2000, 1, 0, 0, 33, 21, 10, 97, 0, 6, 8, 97, 50, 50, 0, 60, 0, 60, 67, 52, 0, 27, 10, 0, 144, 278]
games.append(isn_ope)

# Ivashka vs Andreozzi (1R)
iva_and = [1000, 1, 0, 0, 25, 27, 96, 88, 83, 219, 80, 70, 0, 0, 50, 22, 58, 40, 65, 59, -4, -10, 0, 0, 132, 338]
games.append(iva_and)

# Norrie vs Aliassime (1R)
ali_nor = [1000, 1, 0, 0, 23, 18, 48, 58, 24, 27, 48, 58, 0, 0, 60, 59, 70, 60, 73, 63, 47, 47, 10, 25, 164, 224]
games.append(ali_nor)

# Dzumhur vs Ramos-Viñolas (1R)
dzu_ram = [1000, 1, 0, 0, 26, 31, 54, 91, 129, 61, 23, 17, 0, 100, 25, 46, 0, 0, 52, 49, -13, -29, 25, 25, 171, 212]
games.append(dzu_ram)

# Karlovic vs Ebden (1R)
kar_ebd = [1000, 1, 0, 0, 40, 31, 89, 49, 55, 173, 14, 39, 33, 67, 50, 29, 63, 25, 56, 60, 12, -6, 10, 10, 185, 193]
predict.append(kar_ebd)

# Kohlschreiber vs Herbert (1R)
koh_her = [1000, 1, 0, 0, 35, 27, 39, 44, 91, 39, 16, 36, 80, 20, 50, 67, 50, 67, 58, 58, -13, 20, 180, 90, 164, 224]
predict.append(koh_her)

# Thompson vs Delbonis (1R)
tho_del = [1000, 1, 0, 0, 24, 28, 77, 80, 68, 66, 60, 33, 0, 0, 54, 54, 50, 60, 63, 46, -6, 0, 0, 25, 156, 242]
predict.append(tho_del)

# Shapovalov vs Tiafoe (QF)
sha_tia = [1000, 5, 0, 1, 19, 21, 23, 34, 41, 25, 23, 29, 50, 50, 68, 47, 73, 50, 64, 59, 15, 13, 180, 180, 155, 249]
predict.append(sha_tia)

# Isner vs Aliassime (SF)
isn_ali = [1000, 6, 1, 0, 33, 18, 9, 57, 23, 26, 8, 57, 0, 0, 70, 70, 71, 80, 67, 63, 10, 48, 1000, 0, 158, 239]
predict.append(isn_ali)

nasos = []
for game in games:
    nas = []
    nas.append(game[6] / 1000)
    nas.append(game[7] / 1000)
    nas.append(game[18] / 100)
    nas.append(game[19] / 100)
    nasos.append(nas)

juasos = []
for predict_item in predict:
    juas = []
    juas.append(predict_item[6] / 1000)
    juas.append(predict_item[7] / 1000)
    juas.append(predict_item[18] / 100)
    juas.append(predict_item[19] / 100)
    juasos.append(juas)

#np.set_printoptions(suppress=True)
X = np.array(nasos)
y = np.array([[0, 1], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]])
Z = np.array(juasos)
sol = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])

index = 0

for e in X:
    print("X:", e, "Sol:", y[index], "Network:", nn.predict(e))
    index += 1

nn.fit(X, y, learning_rate = 0.03, epochs = num_epochs)

index = 0

for e in X:
    '''
    coef = nn.predict(e)
    prob_opp = round((coef[0] + 1) * 100 / 2, 2)
    prob_fav = round(100 - prob_opp, 2)
    inv_odd1 = 1 / games[index][18]
    inv_odd2 = 1 / games[index][19]
    inv_tot = inv_odd1 + inv_odd2
    prob_bookmark1 = inv_odd1 * (100 - (inv_tot * 100 - 100))
    prob_bookmark2 = inv_odd2 * (100 - (inv_tot * 100 - 100))
    new_odd1 = round(100 / prob_bookmark1, 2)
    new_odd2 = round(100 / prob_bookmark2, 2)
    value1 = round((new_odd1 * prob_fav / 100 - 1) * 100, 2)
    value2 = round((new_odd2 * prob_opp / 100 - 1) * 100, 2)
    '''

    #if (predict[index][18] >= 1.50 and predict[index][18] <= 2.50 and value1 > 0):
    #print("Pick pel favorit del partit " + games_names[index] + ":")
    #print("Z:", e, "Sol:", y[index], "Network:", coef, "Prob:", str(prob_fav) + "%", "-", str(prob_opp) + "%", "Value:", str(value1), "-", str(value2))
    #else:
    #if (predict[index][19] >= 1.50 and predict[index][19] <= 2.50 and value2 > 0):
    #print("Pick per la sorpresa del partit " + games_names[index] + ":")
    #print("Z:", e, "Sol:", sol[index], "Network:", coef, "Prob:", str(prob_fav) + "%", "-", str(prob_opp) + "%", "Value:", str(value1), "-", str(value2))
    print("X:", e, "Sol:", y[index], "Network:", nn.predict(e))
    index += 1

index = 0

for e in Z:
    print("Z:", e, "Sol:", sol[index], "Network:", nn.predict(e))
    index += 1

# Gràfica
plt.figure("The Beast Training")
plt.title("Test 05/04/2019")

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
