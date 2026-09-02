from sentence_transformers import SentenceTransformer

from app.config import settings


def main() -> None:
    model = SentenceTransformer(settings.embedding_model, local_files_only=False)
    dimensions = model.get_sentence_embedding_dimension()
    print(f"Embedding model ready: {settings.embedding_model} ({dimensions} dimensions)")


if __name__ == "__main__":
    main()
