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
            return index

    if isTE:
        return searchKeyDictionaryFromTE(matrix, key, value)

    return False

def searchKeyDictionaryFromTE(matrix, key, value):
    for index, item in matrix.items():
        # 1st case
        value = value.replace(" Del ", " del ")

        if item[key] == value:
            return index
        else:
            explode = value.split(" ")

            # 2nd case
            if len(explode) == 3:
                new_value = explode[0] + " " + explode[2]

                if item[key] == new_value:
                    return index

            # 3rd case
            new_value = value.replace("-", " ")

            if item[key] == new_value:
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
                    return index

            # 5th case
            replace_dict = {"Julien": "Julian", "Marco": "Marko", "Brinkman": "Brinkmann", "Flavius": "flavius", "Samuel": "Sam"}
            new_value = value

            for orig_string, new_string in replace_dict.items():
                new_value = new_value.replace(orig_string, new_string)

            if item[key] == new_value:
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
                    return index

            # 7th case
            explode = value.split(" ")

            if len(explode) == 4:
                new_value = explode[1] + " " + explode[2] + " " + explode[3]

                if item[key] == new_value:
                    return index

            # 8th case
            if len(explode) == 3:
                new_value = explode[0] + "-" + explode[1] + " " + explode[2]

                if item[key] == new_value:
                    return index
    return False
