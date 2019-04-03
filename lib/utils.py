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

def replaceMultiples(mainString, toBeReplaces, newTuple):
    # Iterate over the strings to be replaced
    for index, elem in enumerate(toBeReplaces) :
        # Check if string is in the main string
        if elem in mainString :
            # Replace the string
            mainString = mainString.replace(elem, newTuple[index])

    return  mainString

def search_td(matrix, key1, value1, key2, value2):
    for index, dictionary in enumerate(matrix):
        if dictionary[key1] == value1 and dictionary[key2] == value2:
            return index

    return -1

def searchKeyDictionaryByValue(matrix, key, value, isTE):
    for index, item in matrix.items():
        if item[key] == value:
            print ("Case 0")
            return index

    if isTE:
        return searchKeyDictionaryFromTE(matrix, key, value)

    return False

def searchKeyDictionaryFromTE(matrix, key, value):
    for index, item in matrix.items():
        item[key] = item[key].replace("  ", " ")

        # 1st case
        value = value.replace(" Del ", " del ")

        if item[key] == value:
            print("Case 1")
            return index
        else:
            explode = value.split(" ")

            # 2nd case
            if len(explode) == 3:
                new_value = explode[0] + " " + explode[2]

                if item[key] == new_value:
                    print("Case 2")
                    return index

            # 3rd case
            new_value = value.replace("-", " ")

            if item[key] == new_value:
                print("Case 3")
                return index

            # 4th case
            partValue = value.split("-")

            if len(partValue) == 2:
                new_value = partValue[0]
                partValueBySpace = partValue[1].split(" ")

                for index2, part in enumerate(partValueBySpace):
                    if index2 > 0:
                        new_value += " " + part

                if item[key] == new_value:
                    print("Case 4")
                    return index

            # 5th case
            replace_dict = {"Julien": "Julian", "Marco": "Marko", "Brinkman": "Brinkmann", "Flavius": "flavius", "Samuel": "Sam", "Brendan": "Brendon", "Joshua": "Josh", "Matt": "Matthew", "Philipp": "Philip", "Alexander": "Aliaksandr", "Vladzimir": "Vladimir", "Segey": "Sergey", "Aleksandr": "Aliaksandr", "loic": "Loic", "Sant'Anna": "Santanna", "Vinícius": "Vinicius", "Aleksandar": "Alexandar", "Zack": "Zachary", "McNicol": "Mcnicol", "Tianjia": "Tian jia", "Weiqiang": "Wei Qiang", "Ruixuan": "Rui-Xuan", "Cortes": "Cortez", "Franco": "Franko", "Roko": "Rocco", "Al ": "Haitham ", "Tareq": "Tarek", "Jean Baptiste": "Jean-baptiste", "Giorgos": "George"}
            new_value = value

            for orig_string, new_string in replace_dict.items():
                new_value = new_value.replace(orig_string, new_string)

            if item[key] == new_value\
               or item[key] == new_value.replace("Marko", "Marco")\
               or item[key] == new_value.replace("Aliaksandr", "Alexander")\
               or item[key] == new_value.replace("Aliaksandr", "Alexandros"):
                print("Case 5")
                return index

            # 6th case
            pos_open_bracket = value.find("(")
            first_end = pos_open_bracket - 1
            pos_close_bracket = value.find(")")
            second_start = pos_close_bracket + 1

            if pos_open_bracket > -1 and pos_close_bracket > -1:
                #There is a number between brackets
                value = value[:first_end] + value[second_start:]

                if item[key] == value:
                    print("Case 6")
                    return index

            # 7th case
            explode = value.split(" ")

            if len(explode) == 4:
                new_value = explode[1] + " " + explode[2] + " " + explode[3]

                if item[key] == new_value:
                    print("Case 7")
                    return index

            # 8th case
            if len(explode) == 3:
                new_value = explode[0] + "-" + explode[1] + " " + explode[2]

                if item[key] == new_value:
                    print("Case 8")
                    return index

            # 9th case
            if len(explode) == 3:
                new_value = explode[0] + " " + explode[2] + " " + explode[1]

                if item[key] == new_value:
                    print("Case 9")
                    return index

            # 10th case
            value = value.replace(" De ", " de ")

            if item[key] == value:
                print("Case 10")
                return index

            # 11th case
            value = value.replace(" de ", " De ")

            if item[key] == value:
                print("Case 11")
                return index

    for index, item in matrix.items():
        # 12th case
        if len(explode) > 2:
            new_value = explode[0] + " " + explode[len(explode) - 1]

            if item[key] == new_value:
                print("Case 12")
                return index

        # 13th case
        if len(explode) > 2:
            new_value = explode[0] + " " + explode[1]

            if item[key] == new_value:
                print("Case 13")
                return index

        # 14th case
        if len(explode) > 2:
            new_value = explode[0] + " " + explode[1] + " " + explode[len(explode) - 1]

            if item[key] == new_value:
                print("Case 14")
                return index

        # 15th case
        explode_item = item[key].split(" ")

        if len(explode_item) > 2:
            new_key = explode_item[0] + " " + explode_item[len(explode_item) - 1]

            if new_key == value:
                print("Case 15")
                return index

        # 16th case
        if len(explode) == 3:
            new_value = explode[0] + " " + explode[1] + "-" + explode[2]

            if item[key] == new_value:
                print("Case 16")
                return index

        # 17th case
        if len(explode_item) == 3:
            new_key = explode_item[0] + " " + explode_item[1]

            if new_key == value:
                print("Case 17")
                return index

        # 18th case
        if len(explode_item) == 2 and "-" in explode_item[0]:
            firstname = explode_item[0].split("-")
            new_key = firstname[0] + " " + explode_item[1]

            if new_key == value:
                print("Case 18")
                return index

        # 19th case
        if len(explode) == 4:
            new_value = explode[0] + " " + explode[1] + " " + explode[2]

            if item[key] == new_value:
                print("Case 19")
                return index

        # 20th case
        if len(explode_item) == 2 and "-" in explode_item[0] and "-" in explode_item[1]:
            firstname = explode_item[0].split("-")
            lastname = explode_item[1].split("-")
            new_key = firstname[0] + " " + lastname[0]

            if new_key == value:
                print("Case 20")
                return index

        # 21th case
        if len(explode) == 4:
            new_value = explode[0] + " " + explode[2]

            if item[key] == new_value:
                print("Case 21")
                return index

    return False
