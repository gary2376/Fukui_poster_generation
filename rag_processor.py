import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import spacy
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import torch
from nltk.corpus import wordnet
import nltk

# 📌 下載 WordNet 資料
nltk.download('wordnet')

# 📌 設定 SQLite & FAISS
DB_PATH = "E:\python_project\contest\poster_fukui\code\\full_function\DB\\rag_data.db"
FAISS_INDEX_PATH = "E:\python_project\contest\poster_fukui\code\\full_function\DB\\faiss_index.bin"

# 📌 初始化 SBERT
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 📌 載入 FAISS Index
index = faiss.read_index(FAISS_INDEX_PATH)

# 📌 過濾低價值詞
def filter_irrelevant_words(word_list):
    """
    過濾掉無意義的形容詞與名詞，如數量詞、泛指詞、模糊詞彙。
    """
    STOPWORDS = {
        "few", "such", "many", "little", "more", "some", "other", "several", "certain", "much", "bit",
        "people", "thing", "way", "year", "day", "time", "someone", "everyone", "anything", "nothing",
        "everything", "everybody", "nobody", "something", "anyone", "person", "place","main", "seated",
        "closed"
    }
    return [word for word in word_list if word.lower() not in STOPWORDS and len(word) >= 5]

# 📌 RAG 查詢（使用 FAISS + SQLite）
def search_rag(query, top_k=3):
    query_vector = sbert_model.encode(query, convert_to_numpy=True).reshape(1, -1)

    # 使用 FAISS 查詢最近鄰
    _, indices = index.search(query_vector, top_k)

    # 連接 SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT location, translated_comment, keyword, adj FROM reviews")
    results = cursor.fetchall()
    conn.close()

    # 返回最相關的結果
    return [
        {
            "location": results[idx][0],
            "translated_comment": results[idx][1],
            "keyword": results[idx][2],
            "adj": results[idx][3]
        }
        for idx in indices[0]
    ]

# 📌 找出最重要的關鍵字 & 形容詞
def get_most_important_words(results, column, top_n=5):
    all_words = []
    for entry in results:
        all_words.extend(entry[column].split(", "))
    word_counts = Counter(all_words)
    return filter_irrelevant_words([word for word, _ in word_counts.most_common(top_n)])

# 📌 載入 CLIP 模型
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
nlp = spacy.load("en_core_web_sm")

# 📌 CLIP 進行視覺相關性評估
def get_sentence_visual_score(sentence):
    text_inputs = clip_processor(text=[sentence], return_tensors="pt", truncation=True, max_length=77)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
    return text_features.norm(dim=-1).item()

# 📌 提取名詞與形容詞
def extract_visual_words(sentence):
    doc = nlp(sentence)
    words = {"NOUN": [], "ADJ": []}
    for token in doc:
        if token.pos_ in {"NOUN", "ADJ"} and not token.is_stop:
            words[token.pos_].append(token.text.lower())
    return words

# 📌 過濾非物理實體
VALID_HYPERNYMS = {
    "physical_entity.n.01", "natural_object.n.01", "artifact.n.01",
    "location.n.01", "body_of_water.n.01", "geological_formation.n.01",
    "structure.n.01", "object.n.01"
}

def is_physical_entity(word):
    synsets = wordnet.synsets(word, pos=wordnet.NOUN)
    if not synsets:
        return True
    for synset in synsets:
        hypernyms = synset.hypernyms()
        if any(hyp.name().startswith(tuple(VALID_HYPERNYMS)) for hyp in hypernyms):
            return True
    return False

# 📌 RAG 查詢 + CLIP 過濾
def process_query(query):
    retrieved_results = search_rag(query)

    top_keywords = get_most_important_words(retrieved_results, "keyword", top_n=3)
    top_adjectives = get_most_important_words(retrieved_results, "adj", top_n=3)

    final_nouns = set()
    final_adjs = set()

    for res in retrieved_results:
        score = get_sentence_visual_score(res["translated_comment"])
        if score > 0.5:
            words = extract_visual_words(res["translated_comment"])
            filtered_nouns = [word for word in words["NOUN"] if is_physical_entity(word)]
            final_nouns.update(filter_irrelevant_words(filtered_nouns))
            final_adjs.update(filter_irrelevant_words(words["ADJ"]))

    return {
        "keywords": list(set(top_keywords)),
        "adjectives": list(set(top_adjectives)),
        "related_nouns": list(final_nouns),
        "related_adjectives": list(final_adjs)
    }
