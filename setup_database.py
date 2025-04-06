import sqlite3
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# **📌 設定 SQLite 資料庫**
DB_PATH = "rag_data.db"

# **📌 讀取 CSV 檔案**
csv_file_1 = "E:\\python_project\\contest\\poster_fukui\\google_comment\\google_new_comment_1\\all_tripadviser_en_nlp.csv"
csv_file_2 = "E:\\python_project\\contest\\poster_fukui\\google_comment\\google_new_comment\\all_jalan_en_nlp.csv"

df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)
df = pd.concat([df1, df2], ignore_index=True)

# **📌 初始化 Sentence-BERT**
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# **📌 生成語義向量**
df["embeddings"] = df["translated_comment"].apply(lambda x: sbert_model.encode(x, convert_to_numpy=True))
dimension = df["embeddings"][0].shape[0]

# **📌 建立 FAISS 向量索引**
index = faiss.IndexFlatL2(dimension)
embeddings_matrix = np.vstack(df["embeddings"].to_numpy())
index.add(embeddings_matrix)

# **📌 儲存 FAISS Index**
faiss.write_index(index, "faiss_index.bin")

# **📌 連接 SQLite**
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# **📌 建立資料表**
cursor.execute('''
    CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        location TEXT,
        translated_comment TEXT,
        keyword TEXT,
        adj TEXT
    )
''')

# **📌 插入資料**
df[["location", "translated_comment", "keyword", "adj"]].to_sql("reviews", conn, if_exists="replace", index=False)

# **📌 確保變更儲存**
conn.commit()
conn.close()

print("✅ CSV 資料已存入 SQLite，FAISS Index 已建立")
