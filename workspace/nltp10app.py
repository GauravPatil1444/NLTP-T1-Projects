!pip install streamlit
!pip install pyspellchecker
import streamlit as st
from transformers import pipeline
from spellchecker import SpellChecker
import nltk
import threading
from queue import Queue



try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

spell = SpellChecker()
text_gen = pipeline("text-generation", model="gpt2")

def check_spelling(text, result_queue):
    words = text.split()
    misspelled = spell.unknown(words)
    result_queue.put(list(misspelled))

def correct_spelling(text, result_queue):
    words = text.split()
    corrected_words = [spell.correction(word) if word in spell.unknown([word]) else word for word in words]
    result_queue.put(' '.join(corrected_words))

def autocomplete_text(prefix, max_length, result_queue):
    try:
        completions = text_gen(prefix, max_length=max_length, num_return_sequences=1, do_sample=True)
        result_queue.put(completions[0]['generated_text'])
    except Exception as e:
        result_queue.put(f"Error during autocompletion: {e}")

def main():
    """Main function to run the Streamlit app."""
    st.title("NLTP Spelling and Autocomplete Tool")

    # Input text area
    input_text = st.text_area("Enter your text:", height=200)

    # Buttons for actions
    col1, col2, col3 = st.columns(3)
    result_queue = Queue()

    if col1.button("Check Spelling"):
        if not input_text:
            st.warning("Please enter text.")
        else:
            thread = threading.Thread(target=check_spelling, args=(input_text, result_queue))
            thread.start()
            thread.join()  # Wait for the thread to finish
            result = result_queue.get()
            st.subheader("Misspelled Words:")
            st.write(", ".join(result))

    if col2.button("Correct Spelling"):
        if not input_text:
            st.warning("Please enter text.")
        else:
            thread = threading.Thread(target=correct_spelling, args=(input_text, result_queue))
            thread.start()
            thread.join()
            result = result_queue.get()
            st.subheader("Corrected Text:")
            st.write(result)

    if col3.button("Autocomplete Text"):
        if not input_text:
            st.warning("Please enter text.")
        else:
            thread = threading.Thread(target=autocomplete_text, args=(input_text, 50, result_queue))
            thread.start()
            thread.join()
            result = result_queue.get()
            st.subheader("Autocomplete Result:")
            st.write(result)

if __name__ == "__main__":
   
    main() #changed to call main() directly
   
