import os

import streamlit as st

from faiss_utils import load_index, run_query
from openai_utils import response_openai

# 定数定義
USER_NAME = "user"
ASSISTANT_NAME = "assistant"
INDEX_PATH = ".\\database\\index\\index.faiss"

# チャットログを保存したセッション情報を初期化
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# Streamlit アプリのタイトルを設定
st.title("GPT-4o-mini Chat Application")

# ユーザーのメッセージ入力
user_msg = st.chat_input("ここにメッセージを入力")

# 以前のチャットログを表示
for chat in st.session_state.chat_log:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])


with st.sidebar:
    use_rag = st.checkbox("RAGを使用する", value=False)

if user_msg:
    try:
        # 最新のメッセージを表示
        with st.chat_message(USER_NAME):
            st.write(user_msg)

            # RAGを使用する場合
            rag_text_data = ""
            if use_rag:
                # インデックスが未生成の場合はエラーメッセージを表示して終了
                if not os.path.exists(INDEX_PATH):
                    st.error("RAG用のデータが未アップロードです。")
                    st.stop()

                # インデックスを取得
                index = load_index()

                # クエリを実行して結果を取得。user_msgに追加。
                rag_text_data = run_query(user_msg, index)

        # セッションにチャットログを追加
        st.session_state.chat_log.append({"role": USER_NAME, "content": user_msg})
        messages = st.session_state.chat_log.copy()

        # RAG用のデータがある場合はメッセージに追加
        if len(rag_text_data) > 0:
            messages.append(
                {
                    "role": USER_NAME,
                    "content": "以下の情報を利用して回答してください。"
                    + "\n".join(rag_text_data),
                }
            )

        # アシスタントのメッセージを取得して表示
        assistant_msg = response_openai(messages)
        with st.chat_message(ASSISTANT_NAME):
            st.write(assistant_msg)
            with st.expander("RAG利用データ"):
                st.write(rag_text_data)

        # セッションにアシスタントのメッセージを追加
        st.session_state.chat_log.append(
            {"role": ASSISTANT_NAME, "content": assistant_msg}
        )
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        st.stop()
