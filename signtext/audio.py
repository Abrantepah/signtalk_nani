
import os
import cv2
from faster_whisper import WhisperModel
import tempfile
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import pandas as pd
import nltk
import pickle
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS, Chroma
from nltk.stem import WordNetLemmatizer

# === Initialize model once at module level ===
whisper_model = WhisperModel("base", device="cpu")


# === Step 1: Setup ===
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Embedding model
embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Load FAISS (sentence-level) and Chroma (word-level) vectorstores
vectorstore_sentence = FAISS.load_local("sign_retrieval_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
vectorstore_words = Chroma(persist_directory="sign_retrieval_chroma", embedding_function=embedding_model)

# === Step 2: Text Preprocessing ===
lemmatizer = WordNetLemmatizer()
auxiliary_verbs = {"am", "is", "are", "was", "were"}

def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    cleaned = []
    for word, tag in nltk.pos_tag(words):
        if word in auxiliary_verbs:
            continue
        if tag.startswith('V'):
            lemma = lemmatizer.lemmatize(word, pos='v')
        else:
            lemma = lemmatizer.lemmatize(word)
        cleaned.append(lemma)
    return cleaned

# === Step 3: Video Retrieval Function ===
def retrieve_video(user_input: str, similarity_threshold: float = 0.8):
    results = vectorstore_sentence.similarity_search_with_score(user_input, k=1)
    if results:
        doc, score = results[0]
        similarity = 1 - score
        print(f"🧠 Sentence Similarity: {similarity:.3f}")
        if similarity >= similarity_threshold:
            return [doc.metadata.get("response", "No Video ID linked.")]
    
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
                video_ids.append(doc.metadata.get("response", "No VideoID found"))
            else:
                video_ids.append("No VideoID found")
        else:
            video_ids.append("No VideoID found")

    return video_ids

# # === Step 4: Audio Recording + Transcription ===
# def record_audio(duration=7, sample_rate=16000):
#     print(f"🎙️ Recording for {duration} seconds...")
#     audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
#     sd.wait()
#     return sample_rate, audio

def transcribe_audio_with_whisper(sr, audio, model):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        wav.write(tmpfile.name, sr, (audio * 32767).astype(np.int16))
        print("🧠 Transcribing...")
        result = model.transcribe(tmpfile.name)
        return result["text"]



# === Step 5: Run Once ===
def process_audio_file(file_path):
    print(f"📥 Processing audio file: {file_path}")

    # Transcription
    segments, info = whisper_model.transcribe(file_path)

    # Concatenate all transcribed segments into one text string
    text = " ".join([segment.text for segment in segments])

    print("📝 Transcribed Text:\n", text)

    # Video retrieval
    video_ids = retrieve_video(text)
    print("🎬 Suggested Video IDs:", video_ids)

    return text, video_ids

