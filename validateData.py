from cassandra.cluster import Cluster
import argparse
from operator import attrgetter
from datetime import datetime, timedelta

# Variables
parser = argparse.ArgumentParser()
parser.add_argument("table")
parser.add_argument("field")
args = parser.parse_args()
table = args.table
field = args.field
'''
dates = []
current_date = "2013-09-30"
last_date = "2017-12-25"

# Dates
while current_date <= last_date:
    dates.append(current_date)
    params = current_date.split("-")
    current_datetime = datetime(int(params[0]), int(params[1]), int(params[2]))
    current_datetime += timedelta(days=7)
    current_date = current_datetime.strftime("%Y-%m-%d")
'''

# Select
cluster = Cluster(["127.0.0.1"])
session = cluster.connect("beast")
'''
# Count ranking by date
for date in dates:
    query = "SELECT * FROM " + table + " WHERE " + field + " = '" + date + "'"
    rows = session.execute(query)
    count = 0

    for row in rows:
        count += 1

    print(date + ": " + str(count))

exit()
'''
query = "SELECT * FROM " + table
rows = session.execute(query)
sorted_rows = sorted(rows, key=attrgetter(field))
print("Valor mínim de " + field + ": " + str(getattr(sorted_rows[0],field)))
print("Valor màxim de " + field + ": " + str(getattr(sorted_rows[len(sorted_rows) - 1],field)))

# Close connections
session.shutdown()
cluster.shutdown()
