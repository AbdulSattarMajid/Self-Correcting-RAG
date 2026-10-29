import pickle

from rank_bm25 import BM25Okapi
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore

from common import STORAGE_DIR, BM25_PATH, tokenize


def main():
    print("Loading local document store...")
    docstore = SimpleDocumentStore.from_persist_dir(STORAGE_DIR)
    all_nodes = list(docstore.docs.values())

    # Index LEAF nodes only, so BM25 and Chroma search the same 256-token chunks.
    leaf_nodes = get_leaf_nodes(all_nodes)
    print(f"Found {len(all_nodes)} nodes, using {len(leaf_nodes)} leaf nodes for BM25.")

    leaf_ids, leaf_texts, leaf_parent_ids = [], [], []
    parent_texts = {}

    for leaf in leaf_nodes:
        leaf_ids.append(leaf.node_id)
        leaf_texts.append(leaf.get_content())

        # Root-level nodes with no parent act as their own parent.
        parent_id = leaf.parent_node.node_id if leaf.parent_node else leaf.node_id
        leaf_parent_ids.append(parent_id)

        if parent_id not in parent_texts:
            parent_texts[parent_id] = docstore.get_document(parent_id).get_content()

    print("Building BM25 Lexical Index...")
    tokenized_corpus = [tokenize(text) for text in leaf_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    index_data = {
        "bm25": bm25,
        "leaf_ids": leaf_ids,
        "leaf_texts": leaf_texts,
        "leaf_parent_ids": leaf_parent_ids,
        "parent_texts": parent_texts,
    }

    with open(BM25_PATH, "wb") as f:
        pickle.dump(index_data, f)

    print(f"✅ BM25 Keyword Index built and saved ({len(leaf_ids)} leaves, {len(parent_texts)} parents).")


if __name__ == "__main__":
    main()