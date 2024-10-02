import chromadb
import numpy as np
import spacy
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Initialize models
knowledge_model = SentenceTransformer('all-mpnet-base-v2')  # Knowledge-based model
nlp = spacy.load('en_core_web_md')  # Semantic model (Spacy)

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(
    path="my_vectordb",
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# Access or create a collection (to add/search embeddings)
collection = chroma_client.get_or_create_collection("company_embeddings")


def generate_composite_embedding(row, small_data_flag = False):
    """Generate composite embedding for a single row."""
    try:
        if small_data_flag:
            entity_representation = f"{row['name']}"
        else:
            entity_representation = f"{row['name']} {row['country']} {row['website']} {row['linkedin']}"

        # Knowledge-based embedding
        knowledge_embedding = knowledge_model.encode(entity_representation)

        # Semantic embedding (from Spacy)
        doc = nlp(entity_representation)
        semantic_embedding = doc.vector

        combined_embedding = np.concatenate([knowledge_embedding, semantic_embedding])
        return combined_embedding

    except Exception as e:
        print(f"Error generating embedding for row: {row}, Error: {e}")
        return None


def add_embeddings_to_chroma(df):
    """Generate embeddings for each row and add to ChromaDB."""
    for idx, row in df.iterrows():
        embedding = generate_composite_embedding(row)

        if embedding is not None:
            # Convert to list (ChromaDB expects embeddings as lists, not numpy arrays)
            embedding_list = embedding.tolist()

            # Add the embedding to ChromaDB
            collection.add(
                ids=[row['id']],  # Assuming 'id' is the unique identifier in the dataset
                embeddings=[embedding_list],
                metadatas=[{
                    'name': row['name'],
                    'country': row['country'],
                    'website': row['website'],
                    'linkedin': row['linkedin']
                }]
            )
        else:
            print(f"Skipping row {idx} due to error.")