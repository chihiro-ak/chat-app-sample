import os

import faiss
import numpy as np

from openai_utils import text_to_vector

INDEX_PATH = ".\\database\\index\\index.faiss"
SOURCE_PATH = ".\\database\\source"


# インデックスを生成
def make_index():
    text_files = os.listdir(SOURCE_PATH)

    # テキストファイルを読み込み、ベクトル化
    vectors = []
    for text_file in text_files:
        with open(os.path.join(SOURCE_PATH, text_file), "r", encoding="utf-8") as f:
            text_data = f.read()
        vector = text_to_vector(text_data)
        vectors.append(vector)

    # ベクトルをnumpy配列に変換
    vectors_array = np.array(vectors).astype("float32")

    # インデックスを生成
    index = faiss.IndexFlatIP(vectors_array.shape[1])
    index.add(vectors_array)

    # インデックスを保存
    if not os.path.exists(".\database\index"):
        os.makedirs(".\database\index")
    faiss.write_index(index, INDEX_PATH)

    return index


# インデックスをロード
def load_index():
    index = faiss.read_index(INDEX_PATH)
    return index


# クエリを実行
def run_query(query, index):
    try:
        # クエリをベクトル化し、正規化
        query_vector = text_to_vector(query)
        query_vector_array = np.array(query_vector).astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_vector_array)

        # クエリベクトルを使って検索を実行（k=5で上位5件を取得）
        k = 3
        D, I = index.search(query_vector_array, k)

        # 検索でヒットしたものに対し、ファイル名を取得して表示
        text_data = []
        for i in range(k):
            text_file_path = os.path.join(
                SOURCE_PATH,
                f"{os.path.splitext(os.listdir(SOURCE_PATH)[I[0][i]])[0]}.txt",
            )
            with open(text_file_path, "r", encoding="utf-8") as f:
                text_data.append(f.read())

        return text_data

    except Exception as e:
        return f"エラーが発生しました: {e}"
