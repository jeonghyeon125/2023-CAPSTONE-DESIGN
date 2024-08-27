# db 등록
import pandas as pd
import sqlite3


""" 단어 사전 """

# Adjective
df = pd.read_csv("dic/Adjective.csv", encoding = 'cp949')
# connect to database
database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype = {
    "word": "CharField",
    "word_type": "CharField",
    "note" : "CharField"
}
df.to_sql(name = 'team4db_adjective', con = conn, if_exists = 'replace', 
          dtype = dtype, index = True, index_label = "id")
conn.close()

# Adverb
df = pd.read_csv("dic/Adverb.csv", encoding = 'cp949')
database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype = {
    "word": "CharField",
    "word_type": "CharField",
    "note" : "CharField"
}
df.to_sql(name = 'team4db_adverb', con = conn, if_exists = 'replace', 
          dtype = dtype, index = True, index_label = "id")
conn.close()

# dic_usr
df = pd.read_csv("dic/dic_usr.csv", encoding = 'cp949')
database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype = {
    "word": "CharField",
    "word_type": "CharField",
    "note" : "CharField"
}
df.to_sql(name = 'team4db_dic_usr', con = conn, if_exists = 'replace', 
          dtype = dtype, index = True, index_label = "id")
conn.close()

# eomi
df = pd.read_csv("dic/Eomi.csv", encoding = 'cp949')
database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype = {
    "word": "CharField",
    "word_type": "CharField",
    "note" : "CharField"
}
df.to_sql(name = 'team4db_eomi', con = conn, if_exists = 'replace', 
          dtype = dtype, index = True, index_label = "id")
conn.close()

# exclamation
df = pd.read_csv("dic/Exclamation.csv", encoding = 'cp949')
database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype = {
    "word": "CharField",
    "word_type": "CharField",
    "note" : "CharField"
}
df.to_sql(name = 'team4db_exclamation', con = conn, if_exists = 'replace', 
          dtype = dtype, index = True, index_label = "id")
conn.close()

# foreign
df = pd.read_csv("dic/Foreign.csv", encoding = 'cp949')
database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype = {
    "word": "CharField",
    "word_type": "CharField",
    "note" : "CharField"
}
df.to_sql(name = 'team4db_foreign', con = conn, if_exists = 'replace', 
          dtype = dtype, index = True, index_label = "id")
conn.close()

# hanja
df = pd.read_csv("dic/Hanja.csv", encoding = 'cp949')
database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype = {
    "word": "CharField",
    "word_type": "CharField",
    "note" : "CharField"
}
df.to_sql(name = 'team4db_hanja', con = conn, if_exists = 'replace', 
          dtype = dtype, index = True, index_label = "id")
conn.close()

# josa
df = pd.read_csv("dic/Josa.csv", encoding = 'cp949')
database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype = {
    "word": "CharField",
    "word_type": "CharField",
    "note" : "CharField"
}
df.to_sql(name = 'team4db_josa', con = conn, if_exists = 'replace', 
          dtype = dtype, index = True, index_label = "id")
conn.close()

# noun
df = pd.read_csv("dic/Noun.csv", encoding = 'cp949')
database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype = {
    "word": "CharField",
    "word_type": "CharField",
    "note" : "CharField"
}
df.to_sql(name = 'team4db_noun', con = conn, if_exists = 'replace', 
          dtype = dtype, index = True, index_label = "id")
conn.close()

# suffix
df = pd.read_csv("dic/Suffix.csv", encoding = 'cp949')
database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype = {
    "word": "CharField",
    "word_type": "CharField",
    "note" : "CharField"
}
df.to_sql(name = 'team4db_suffix', con = conn, if_exists = 'replace', 
          dtype = dtype, index = True, index_label = "id")
conn.close()

# symbol
df = pd.read_csv("dic/Symbol.csv", encoding = 'cp949')
database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype = {
    "word": "CharField",
    "word_type": "CharField",
    "note" : "CharField"
}
df.to_sql(name = 'team4db_symbol', con = conn, if_exists = 'replace', 
          dtype = dtype, index = True, index_label = "id")
conn.close()

# verb
df = pd.read_csv("dic/Verb.csv", encoding = 'cp949')
database = "db.sqlite3"
conn = sqlite3.connect(database)
dtype = {
    "word": "CharField",
    "word_type": "CharField",
    "note" : "CharField"
}
df.to_sql(name = 'team4db_verb', con = conn, if_exists = 'replace', 
          dtype = dtype, index = True, index_label = "id")
conn.close()