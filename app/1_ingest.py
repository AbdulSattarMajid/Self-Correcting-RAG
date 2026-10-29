import chromadb
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from common import DATA_DIR, CHROMA_DIR, STORAGE_DIR, COLLECTION_NAME

load_dotenv()


def main():
    print("Loading embedding model (CPU)...")
    # Downloads the open-source model locally; it is 100% free.
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = None

    print(f"Reading PDFs from {DATA_DIR} ...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()

    # Split into 1024-token Parents and 256-token Children
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[1024, 256], chunk_overlap=20)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    print(f"Created {len(nodes)} nodes total, {len(leaf_nodes)} leaf (child) nodes.")

    print("Initializing ChromaDB and Document Store...")
    # The docstore keeps ALL nodes (parents + children) so parents can be looked up later.
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    db = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)

    # Only the small child nodes are embedded; Chroma ids == leaf node_ids.
    print("Embedding Child nodes into ChromaDB...")
    VectorStoreIndex(leaf_nodes, storage_context=storage_context, show_progress=True)

    storage_context.persist(persist_dir=STORAGE_DIR)
    print("✅ Ingestion Complete! Data saved to /chroma_db and /local_storage.")


if __name__ == "__main__":
    main()