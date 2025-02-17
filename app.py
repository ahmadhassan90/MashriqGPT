import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
import base64
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set Page Configuration
st.set_page_config(page_title="MashriqGPT - Urdu Poetry", layout="wide")

# Function to Encode Image as Base64 for Background
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Set Background Image (Ensure the image exists in the directory)
image_path = "background.jpg"  # Ensure this file exists
if os.path.exists(image_path):
    encoded_bg = get_base64_image(image_path)
    bg_css = f"""
    <style>
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)),  
                        url("data:image/png;base64,{encoded_bg}") no-repeat center center fixed;
            background-size: cover;
        }}

        /* Darker and Bolder Text for Emphasis */
        h1, h3, p, .stTextInput label {{
            color: #1a1a1a !important;  /* Dark Gray */
            font-weight: bold;
        }}

        /* Adjust Sidebar Text for Better Readability */
        .css-1d391kg {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            color: #333;  /* Dark Gray */
            font-weight: bold;
        }}

        /* Footer */
        .footer {{
            position: fixed;
            bottom: 10px;
            right: 20px;
            background-color: #343a40;
            color: white;
            padding: 10px;
            font-size: 14px;
            font-weight: bold;
            border-radius: 5px;
        }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)
else:
    st.warning("⚠️ Background image not found. Please check the file path.")

# Load Model and Tokenizer
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("poetry_lstm.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()
max_sequence_length = model.input_shape[1]

# Function to Generate Poetry
def generate_poetry(seed_text, num_words=45, temperature=0.7):
    generated_words = seed_text.split()
    
    # First input is padded only once
    tokenized_input = tokenizer.texts_to_sequences([" ".join(generated_words)])[0]
    tokenized_input = pad_sequences([tokenized_input], maxlen=max_sequence_length, padding="pre")

    for _ in range(num_words):
        context_words = generated_words[-5:]  # Always look back 5 words
        tokenized_input = tokenizer.texts_to_sequences([" ".join(context_words)])[0]  # No additional padding

        predicted_logits = model.predict(np.array([tokenized_input]), verbose=0)[0]
        predicted_probs = tf.nn.softmax(predicted_logits / temperature).numpy()
        predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)

        next_word = tokenizer.index_word.get(predicted_index, None)
        if not next_word:
            continue
        generated_words.append(next_word)

    poem_lines = [" ".join(generated_words[i:i+5]) for i in range(0, len(generated_words), 5)]
    return "\n".join(poem_lines)

# Sidebar - Instructions
with st.sidebar:
    st.markdown("### 📜 Instructions")
    st.markdown("""
    1. Enter a starting phrase in **Roman Urdu**.  
    2. Click **Generate Poetry**.  
    3. The AI will generate **Urdu poetry** using LSTM.  
    4. Each line will have **5 words** for better structure.  
    5. Enjoy and share your poetry!  
    """)

# Main Content
st.markdown("<h1 style='text-align: center;'>🌙 MashriqGPT - Urdu Poetry Generator</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Salam!</h3>", unsafe_allow_html=True)
st.markdown(
    """
    <p style='text-align: center; font-weight: bold; color: #333;'>
    I am <b>MashriqGPT</b>, a Roman Urdu poetry bot.  
    Tweet at me a line in <b>Roman Urdu</b>, and I'll complete the rest.  
    Apologies if I don't make sense. Let <b>@AhmadHassan</b> and <b>@IsmailDaniyal</b> know if something doesn't seem right!  
    </p>
    """, unsafe_allow_html=True
)

# Poetry Input & Button
seed_text = st.text_input("Enter a starting phrase:", value=st.session_state.get("seed_text", "Raat ka chand chamak raha"))
if seed_text:
    st.session_state["seed_text"] = seed_text  # Store the user input

num_words = 45  # Fixed words between 40-50



if st.button("Generate Poetry"):
    generated_poem = generate_poetry(seed_text, num_words)
    st.text_area("Generated Poem", generated_poem, height=300)

# Footer
st.markdown(
    """
    <div class="footer">
        Developed by Ahmad Hassan and Ismail Daniyal ❤️ | © 2025 MashriqGPT. All Rights Reserved.
    </div>
    """,
    unsafe_allow_html=True,
)
