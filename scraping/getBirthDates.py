from cassandra.cluster import Cluster
from bs4 import BeautifulSoup
import re
import requests
from datetime import datetime, timedelta
import sys
sys.path.append(r'C:\Users\d_mas\Developer\The Beast\lib')
import utils
from colorama import init, Fore, Back, Style

# Variables
init() # Init colorama
evaluated_players = []
birth_dates = dict()

# Get players from DB
cluster = Cluster(["127.0.0.1"])
session = cluster.connect("beast")

query = "SELECT player_keyword, player_atpwt_id, player_te_name, player_te_url, player_birth, player_hand FROM player_by_atpid"
players = session.execute(query)

for player in players:
    if player.player_atpwt_id not in evaluated_players and (player.player_hand == 2 or player.player_birth == "1900-01-01"):
        evaluated_players.append(player.player_atpwt_id)

        if len(evaluated_players) >= 1000:
            exit()
        else:
            print(Style.BRIGHT + Fore.WHITE + str(len(evaluated_players)) + ") " + player.player_te_name + Style.NORMAL + Fore.CYAN + " (" + player.player_atpwt_id + " ▸ " + player.player_keyword + ")")
            # Get birthdate from Tennis Explorer
            url = "https://www.tennisexplorer.com" + player.player_te_url
            r = requests.get(url)
            data = r.text
            soup = BeautifulSoup(data, "html.parser")
            data = list(soup.select("div.date"))

            try:
                age = data[1].text

                # Hand
                hand = data[len(data) - 1].text.replace("Plays: ", "")
                error_hand = False

                if hand == "right":
                    hand = 0
                else:
                    if hand == "left":
                        hand = 1
                    else:
                        error_hand = True

                if error_hand:
                    print(Style.BRIGHT + Fore.RED + "L'arreplegat d'en " + player.player_te_name + " ens està intentant donar gat per llebre amb la mà de joc\n")
                else:
                    if "Height" in age:
                        age = list(soup.select("div.date"))[2].text

                    birthdate_te = utils.getStringBetweenBrackets(age)

                    if not birthdate_te:
                        print(Style.BRIGHT + Fore.RED + "No em foterà pas el bo d'en " + player.player_te_name)
                        age = list(soup.select("div.date"))[0].text
                        birthdate_te = utils.getStringBetweenBrackets(age)
                        birthdate_te = birthdate_te.split(". ")
                        birthdate_te = datetime(int(birthdate_te[2]), int(birthdate_te[1]), int(birthdate_te[0]))
                        birthdate = birthdate_te.strftime("%Y-%m-%d")
                        print(Style.BRIGHT + Fore.YELLOW + "La data de naixement correcta és " + str(birthdate) + "\n")

                        index = 0
                        rankdates = []
                        query = "SELECT player_rankdate FROM player_by_atpid WHERE player_atpwt_id = '" + player.player_atpwt_id + "'"
                        ranks = session.execute(query)

                        for rank in ranks:
                            rankdates.append(rank.player_rankdate)

                        for rankdate in rankdates:
                            update = "UPDATE player_by_atpid SET player_birth = '" + birthdate + "', player_hand = " + str(hand) + " WHERE player_atpwt_id = '" + player.player_atpwt_id + "' AND player_rankdate = '" + str(rankdate) + "'"

                            if index == 0:
                                print(Style.BRIGHT + Fore.GREEN + update + "\n")

                            session.execute(update)
                            index += 1
                    else:
                        birthdate_te = birthdate_te.split(". ")
                        birthdate_te = datetime(int(birthdate_te[2]), int(birthdate_te[1]), int(birthdate_te[0]))

                        # Get birthdate from ATP World Tour
                        url = "https://www.atptour.com/en/players/" + player.player_keyword + "/" + player.player_atpwt_id + "/overview"
                        r = requests.get(url)
                        data = r.text
                        soup = BeautifulSoup(data, "html.parser")

                        try:
                            birthdate_atpwt = list(soup.select("span.table-birthday"))[0].text.strip().replace("(", "").replace(")", "")
                            birthdate_atpwt = birthdate_atpwt.split(".")
                            birthdate_atpwt = datetime(int(birthdate_atpwt[0]), int(birthdate_atpwt[1]), int(birthdate_atpwt[2]))

                            # Birthdate and hand validation
                            if birthdate_te == birthdate_atpwt:
                                # All OK
                                index = 0
                                birthdate = birthdate_atpwt.strftime("%Y-%m-%d")
                                rankdates = []
                                query = "SELECT player_rankdate FROM player_by_atpid WHERE player_atpwt_id = '" + player.player_atpwt_id + "'"
                                ranks = session.execute(query)

                                for rank in ranks:
                                    rankdates.append(rank.player_rankdate)

                                for rankdate in rankdates:
                                    update = "UPDATE player_by_atpid SET player_birth = '" + birthdate + "', player_hand = " + str(hand) + " WHERE player_atpwt_id = '" + player.player_atpwt_id + "' AND player_rankdate = '" + str(rankdate) + "'"

                                    if index == 0:
                                        print(Style.BRIGHT + Fore.GREEN + update + "\n")

                                    session.execute(update)
                                    index += 1
                            else:
                                # Birthdates are different
                                birthdate = birthdate_atpwt.strftime("%Y-%m-%d")
                                print(Style.BRIGHT + Fore.RED + "Irra amb les dates de naixement de l'avi " + player.player_te_name)
                                print(Style.BRIGHT + Fore.YELLOW + "La data de naixement correcta és " + str(birthdate) + "\n")

                                index = 0
                                rankdates = []
                                query = "SELECT player_rankdate FROM player_by_atpid WHERE player_atpwt_id = '" + player.player_atpwt_id + "'"
                                ranks = session.execute(query)

                                for rank in ranks:
                                    rankdates.append(rank.player_rankdate)

                                for rankdate in rankdates:
                                    update = "UPDATE player_by_atpid SET player_birth = '" + birthdate + "', player_hand = " + str(hand) + " WHERE player_atpwt_id = '" + player.player_atpwt_id + "' AND player_rankdate = '" + str(rankdate) + "'"

                                    if index == 0:
                                        print(Style.BRIGHT + Fore.GREEN + update + "\n")

                                    session.execute(update)
                                    index += 1
                        except IndexError:
                                # ATP birthdate is not introduced
                                birthdate = birthdate_te.strftime("%Y-%m-%d")
                                print(Style.BRIGHT + Fore.RED + "Els mindundis d'ATP World Tour no han introduït la data de naixement del bo d'en " + player.player_te_name)
                                print(Style.BRIGHT + Fore.YELLOW + "La data de naixement correcta és " + str(birthdate) + "\n")

                                index = 0
                                rankdates = []
                                query = "SELECT player_rankdate FROM player_by_atpid WHERE player_atpwt_id = '" + player.player_atpwt_id + "'"
                                ranks = session.execute(query)

                                for rank in ranks:
                                    rankdates.append(rank.player_rankdate)

                                for rankdate in rankdates:
                                    update = "UPDATE player_by_atpid SET player_birth = '" + birthdate + "', player_hand = " + str(hand) + " WHERE player_atpwt_id = '" + player.player_atpwt_id + "' AND player_rankdate = '" + str(rankdate) + "'"

                                    if index == 0:
                                        print(Style.BRIGHT + Fore.GREEN + update + "\n")

                                    session.execute(update)
                                    index += 1
            except IndexError:
                print(Style.BRIGHT + Fore.RED + "Béééééffffff, no vull ser en java-príblems de " + player.player_te_name + ", sniffffff\n")
# Close connections
session.shutdown()
cluster.shutdown()
