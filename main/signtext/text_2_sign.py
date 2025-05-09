import os
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
import nltk
from nltk.stem import WordNetLemmatizer
from langchain.vectorstores import Chroma
from langchain.schema import Document

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Load embedding model (BAAI)
embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
# Load your prompt-response CSV (sentence + video ID)
df = pd.read_csv(r"C:\Users\Idan\Desktop\Projects\signtalk_nani\main\staticfiles\final_prompt_response.csv")  # must have "prompt" and "response" columns

# Convert each row into a Langchain Document
docs = [
    Document(
        page_content=str(row["prompt"]),
        metadata={"response": row["response"]}
    )
    for _, row in df.iterrows()
]

# Embed and save to FAISS
vectordb = FAISS.from_documents(docs, embedding=embedding_model)
vectordb.save_local("sign_retrieval_index")

# Load the FAISS vectorstore (make sure allow_dangerous_deserialization=True)
vectorstore_sentence = FAISS.load_local(
    "sign_retrieval_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)



# Download NLTK resources (only once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# --- 1. Load the embedding model ---
embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# --- 2. Load your CSV ---
df = pd.read_csv(r"C:\Users\Idan\Desktop\Projects\signtalk_nani\main\staticfiles\words_meta.csv")

# --- 3. Create documents ---
docs = [
    Document(
        page_content=str(row["SIGN"]),  # Ensure it's a string
        metadata={"response": str(row["ID"])}
    )
    for _, row in df.iterrows()
]

# --- 4. Create vectorstore ---
vectordb = Chroma.from_documents(docs, embedding=embedding_model, persist_directory="sign_retrieval_chroma")
print("✅ Vector store created successfully!")
vectordb.persist()

# --- 5. Load vectorstore ---
vectorstore_words = Chroma(persist_directory="sign_retrieval_chroma", embedding_function=embedding_model)

# --- 6. Preprocessing: Custom auxiliary verb removal + lemmatization ---
lemmatizer = WordNetLemmatizer()

# Define list of auxiliary verbs
auxiliary_verbs = {"am", "is", "are", "was", "were"}

def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    cleaned = []
    for word, tag in nltk.pos_tag(words):
        if word in auxiliary_verbs:
            continue  # skip auxiliary verbs
        if tag.startswith('V'):  # Verb
            lemma = lemmatizer.lemmatize(word, pos='v')
        else:  # Noun, Adjective, etc.
            lemma = lemmatizer.lemmatize(word)
        cleaned.append(lemma)
    return cleaned


#final function to be used
def retrieve_video(user_input: str, similarity_threshold: float = 0.8):
    """
    Retrieve videos based on user input.

    First attempts full sentence retrieval.
    If fails, falls back to word-by-word retrieval.

    Returns a dictionary:
    - mode: 'sentence' or 'word-by-word'
    - videos: list of video IDs (or messages if not found)
    """
    # Step 1: Sentence search
    results = vectorstore_sentence.similarity_search_with_score(user_input, k=1)

    if results:
        doc, score = results[0]
        similarity = 1 - score
        print(f"🧠 Sentence Similarity: {similarity:.3f}")

        if similarity >= similarity_threshold:
            return {
                "mode": "sentence",
                "videos": [doc.metadata.get("response", "No Video ID linked.")]
            }
    
    # Step 2: Fallback - Word-by-word
    print("⚡ Low similarity. Fallback to word-by-word matching...")
    
    cleaned_words = preprocess_text(user_input)
    print(f"🧹 Cleaned important words: {cleaned_words}")

    video_ids = []

    for word in cleaned_words:
        word_results = vectorstore_words.similarity_search_with_score(word, k=1)

        if word_results:
            doc, score = word_results[0]
            word_similarity = 1 - score
            print(f"🔍 Word '{word}' ➔ Similarity {word_similarity:.3f}")

            if word_similarity >= 0.63:
                video_ids.append(doc.metadata.get("response", ""))
            else:
                video_ids.append("No VideoID found")
        else:
            video_ids.append("No VideoID found")

    return {
        "mode": "word-by-word",
        "videos": video_ids
    }
