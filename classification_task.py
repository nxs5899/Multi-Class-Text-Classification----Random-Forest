import pandas as pd
import pickle
from sqlite3 import Error
import sqlite3

# this database contains the commit messages that must be classified using the model built in the past step.
try:
    conn = sqlite3.connect("myWork.db")
except Error as e:
    print(e)

df = pd.read_sql_query('SELECT * FROM cleaned', conn)
df.fillna('.', inplace=True)
df['classed'] = None
# print(df.head())
strings = df['commessage']

pickle_load = open('RandomForest.pickle', 'rb')
clf = pickle.load(pickle_load)

def classify():
    list = []
    for message in strings:
        message1 = [str(message)]
        pred = clf.predict(message1)
        list.append(pred[0])
    return list

predictions = classify()
df['classed'] = predictions

df.to_sql("production", conn, if_exists="replace")
conn.commit()
