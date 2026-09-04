import os

import httpx
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"


def main() -> None:
    st.set_page_config(page_title="EvidenceRAG", page_icon="ER", layout="wide")
    st.title("EvidenceRAG")

    api_url = st.sidebar.text_input("API URL", configured_api_url()).rstrip("/")
    health = get_json(api_url, "/health")
    if health:
        st.sidebar.json(health)

    left, right = st.columns([0.9, 1.1])
    with left:
        render_upload(api_url)
        render_documents(api_url)
    with right:
        render_ask(api_url)


def render_upload(api_url: str) -> None:
    st.subheader("Upload")
    file = st.file_uploader("PDF", type=["pdf"])
    if st.button("Upload", type="primary", disabled=file is None):
        assert file is not None
        response = httpx.post(
            f"{api_url}/documents",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            timeout=180,
        )
        if response.is_success:
            st.success("Indexed")
            st.json(response.json())
        else:
            st.error(response.text)


def render_documents(api_url: str) -> None:
    st.subheader("Documents")
    documents = get_json(api_url, "/documents") or []
    if not documents:
        st.info("No documents indexed")
        return

    for document in documents:
        with st.expander(f"{document['filename']} | {document['status']}"):
            st.write(f"Document ID: `{document['document_id']}`")
            st.write(f"Pages: {document['page_count']} | Chunks: {document['chunk_count']}")
            if st.button("Delete", key=f"delete-{document['document_id']}"):
                response = httpx.delete(
                    f"{api_url}/documents/{document['document_id']}", timeout=30
                )
                if response.is_success:
                    st.rerun()
                st.error(response.text)


def render_ask(api_url: str) -> None:
    st.subheader("Ask")
    question = st.text_area("Question", height=120)
    if st.button("Ask", type="primary", disabled=len(question.strip()) < 3):
        response = httpx.post(
            f"{api_url}/ask",
            json={"question": question.strip()},
            timeout=180,
        )
        if not response.is_success:
            st.error(response.text)
            return

        payload = response.json()
        st.markdown(payload["answer"])
        st.caption(
            f"{payload['grounding']['status']} | "
            f"support {payload['grounding']['support_score']} | "
            f"{payload['latency_ms']} ms"
        )
        for index, citation in enumerate(payload["citations"], start=1):
            with st.expander(f"Source {index}: {citation['document']} page {citation['page']}"):
                st.write(citation["excerpt"])
                st.json(citation)


def get_json(api_url: str, path: str):
    try:
        response = httpx.get(f"{api_url}{path}", timeout=5)
        if response.is_success:
            return response.json()
    except httpx.HTTPError:
        return None
    return None


def configured_api_url() -> str:
    return os.getenv("EVIDENCE_RAG_API_URL", DEFAULT_API_URL)


if __name__ == "__main__":
    main()
