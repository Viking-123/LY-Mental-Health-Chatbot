import nltk
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import openai

nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pickle
import numpy as np
from keras.models import load_model
model = load_model('model.h5')

import json
import random
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

# Set OpenAI API Key
openai.api_key = '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'

# Load the English to Swahili translation model
eng_swa_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-swc")
eng_swa_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-swc")
eng_swa_translator = pipeline("text2text-generation", model=eng_swa_model, tokenizer=eng_swa_tokenizer)

# Load the Swahili to English translation model
swa_eng_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-swc-en")
swa_eng_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-swc-en")
swa_eng_translator = pipeline("text2text-generation", model=swa_eng_model, tokenizer=swa_eng_tokenizer)

# Language detector function for SpaCy
def get_lang_detector(nlp, name):
    return LanguageDetector()

# Load SpaCy model and add language detection pipe
nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)

# Load intents, words, and classes
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return None

def translate_text_eng_swa(text):
    translated_text = eng_swa_translator(text, max_length=128, num_beams=5)[0]['generated_text']
    return translated_text

def translate_text_swa_eng(text):
    translated_text = swa_eng_translator(text, max_length=128, num_beams=5)[0]['generated_text']
    return translated_text

def openai_fallback_response(msg):
    """Generates a response using OpenAI's API."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": msg}]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "I'm here to help. If you need immediate assistance, please consider reaching out to a mental health professional."

def chatbot_response(msg):
    # Use langdetect to detect language
    try:
        detected_language = detect(msg)
        print(f"Detected language in chatbot_response: {detected_language}")
    except LangDetectException:
        print(f"Error in language detection for message: {msg}")
        detected_language = "unknown"
    
    # If message is short (1 or 2 words), assume it's in English (for fallback)
    if len(msg.split()) <= 2:
        detected_language = "en"

    # Handle detected language
    if detected_language == "en":
        # If detected as English, process as normal
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
        if not res:
            res = openai_fallback_response(msg)
        return res
    elif detected_language == 'sw':
        # If detected as Swahili, translate and process
        translated_msg = translate_text_swa_eng(msg)
        ints = predict_class(translated_msg, model)
        res = getResponse(ints, intents)
        if not res:
            res = openai_fallback_response(translated_msg)
        return translate_text_eng_swa(res)
    else:
        # If language detection is unsure, assume English and try processing
        print(f"Language detection failed or unsupported language: {detected_language}")
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
        if not res:
            res = openai_fallback_response(msg)
        return res

from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print(f"User message: {userText}")

    # Check language and get response
    bot_response = chatbot_response(userText)

    return bot_response

if __name__ == "__main__":
    app.run(debug=True)
