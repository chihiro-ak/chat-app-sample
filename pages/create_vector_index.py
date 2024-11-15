import streamlit as st

from faiss_utils import make_index
from file_utils import save_text, text_from_uploaded_file

st.title("create_vector_index")

st.write("RAG用のデータをアップロードしてください。")

# ファイルアップロードコンポーネント
uploaded_file = st.file_uploader(
    "テキストファイルを選択してください",
    type=["txt", "pdf"],
    accept_multiple_files=False,
)

# アップロードボタンが押された&アップロードファイルがある場合
if st.button("アップロード"):
    if uploaded_file is None:
        st.warning("ファイルが選択されていません。")
        st.stop()

    # アップロードされたファイルからテキストデータを取得
    with st.spinner("ファイルを読み込んでいます..."):
        text = text_from_uploaded_file(uploaded_file)

    # テキストデータを分割し、.database/dataにテキストファイルとして保存
    with st.spinner("ファイルを保存しています..."):
        save_text(text, uploaded_file.name)

    # インデックスの生成
    with st.spinner("インデックスを生成しています..."):
        st.session_state.index = make_index()

    st.success("RAG用のデータをアップロードしました。")
