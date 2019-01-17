#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from cassandra.cluster import Cluster
import argparse
from datetime import datetime, timedelta

# Variables
parser = argparse.ArgumentParser()
parser.add_argument("orig_table")
parser.add_argument("target_table")
args = parser.parse_args()
orig_table = args.orig_table
target_table = args.target_table
columns = []
column_names = []
separator = ", "
new_rows = []

# Select
cluster = Cluster(["127.0.0.1"])
session = cluster.connect("beast")

query = "SELECT * FROM player"
rows = session.execute(query)

for row in rows:
    query = "INSERT INTO player (player_C_career, player_C_season, player_C_year, player_G_career, player_G_season, player_G_year, player_H_career, player_H_season, player_H_year, player_I_career, player_I_season, player_I_year, player_birth, player_rankmax, player_te_name, player_te_url, player_atpwt_id, player_country, player_name, player_race, player_ranking, player_rankdate, player_keyword) VALUES (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, '1900-01-01', -1, 'BLANK', 'BLANK', '" + row.player_atpwt_id + "', '" + row.player_country + "', '" + row.player_name + "', " + str(row.player_race) + ", " + str(row.player_ranking) + ", '" + str(row.player_rankdate) + "', '" + row.player_keyword + "')"
    print(query)
    session.execute(query)
exit()

# Get columns from the target table
query = "SELECT * FROM system_schema.columns WHERE keyspace_name = 'beast' AND table_name = '" + target_table + "'"
schema_columns = session.execute(query)

for schema_column in schema_columns:
    column = dict()
    column['name'] = schema_column.column_name
    column_names.append(column['name'])
    column['type'] = schema_column.type
    columns.append(column)

# Get fields values from the origin table
query = "SELECT " + separator.join(column_names) + " FROM " + orig_table
print(query)
rows = session.execute(query)

for row in rows:
    row = list(row)
    print(row)
    insert = "INSERT INTO " + target_table + " (" + separator.join(column_names) + ") VALUES ("
    '''
    for column in columns:
        if column['type'] == "text" or column['type'] == date:
            insert += "'" + row. + "'"
    '''

# Close connections
session.shutdown()
cluster.shutdown()

