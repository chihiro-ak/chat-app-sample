import os

import pymupdf4llm
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter

SOURCE_PATH = ".\\database\\source"
TEMP_PATH = ".\\database\\temp"


# アップロードされたファイルからテキストデータを取得
# PDFファイルの場合はmarkdown形式に変換して取得
def text_from_uploaded_file(uploaded_file):
    try:
        if uploaded_file is not None:
            # ファイルの内容を読み込む
            # textファイルの場合はそのまま読み込む
            if uploaded_file.type == "text/plain":
                text_data = uploaded_file.read().decode("utf-8")

            # pdfファイルの場合はmarkdown形式に変換して読み込む
            elif uploaded_file.type == "application/pdf":
                # ローカルに保存してから読み込む
                if not os.path.exists(TEMP_PATH):
                    os.makedirs(TEMP_PATH)
                pdf_file_path = os.path.join(TEMP_PATH, uploaded_file.name)
                with open(pdf_file_path, "wb") as f:
                    f.write(uploaded_file.read())
                text_data = pymupdf4llm.to_markdown(pdf_file_path)
                # ローカルに保存したファイルを削除
                os.remove(pdf_file_path)

            else:
                raise ValueError("サポートされていないファイル形式です。")

        return text_data

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")


# テキストデータを分割し、.database\dataにテキストファイルとして保存
def save_text(text_data, file_name):
    splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator="。")
    split_data = splitter.split_text(text_data)

    # フォルダがない場合は作成
    if not os.path.exists(".\\database\\source"):
        os.makedirs(".\\database\\source")

    for i, text in enumerate(split_data):
        text_file_path = os.path.join(
            SOURCE_PATH, f"{os.path.splitext(file_name)[0]}_{i}.txt"
        )
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write(text)
