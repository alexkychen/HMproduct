import pandas as pd
import psycopg2 as pg2

conn = pg2.connect(database='postgres', user='postgres',password='password')

cur = conn.cursor()
conn.close()
