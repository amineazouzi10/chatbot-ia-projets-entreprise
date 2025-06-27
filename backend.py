from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import os
import uuid
import re

from google.cloud import speech
from google.cloud import texttospeech

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
RASA_SERVER_URL = "http://localhost:5005/webhooks/rest/webhook"
STATIC_DIR_NAME = "static_audio"
STATIC_DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), STATIC_DIR_NAME)
if not os.path.exists(STATIC_DIR_PATH):
    os.makedirs(STATIC_DIR_PATH)
    print(f"Created static audio directory: {STATIC_DIR_PATH}")

try:
    stt_client = speech.SpeechClient()
    tts_client = texttospeech.TextToSpeechClient()
    print("Google Cloud STT and TTS clients initialized successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not initialize Google Cloud clients: {e}")


def split_text_preserving_format(text, max_length=200):
    """Split text into smaller chunks while preserving formatting like line breaks."""
    if len(text) <= max_length:
        return [text]

    # First, split by double line breaks (paragraphs)
    paragraphs = text.split('\n\n')
    chunks = []

    for paragraph in paragraphs:
        if len(paragraph) <= max_length:
            chunks.append(paragraph)
        else:
            # Split long paragraphs by single line breaks
            lines = paragraph.split('\n')
            current_chunk = ""

            for line in lines:
                # If this line would make the chunk too long, save current chunk and start new one
                test_chunk = current_chunk + '\n' + line if current_chunk else line

                if len(test_chunk) > max_length and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = line
                else:
                    current_chunk = test_chunk

                # If even a single line is too long, split it by sentences
                if len(current_chunk) > max_length:
                    # Split by sentences while preserving the line structure
                    sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
                    temp_chunk = ""

                    for sentence in sentences:
                        if len(temp_chunk + sentence) > max_length and temp_chunk:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = sentence
                        else:
                            temp_chunk += " " + sentence if temp_chunk else sentence

                    current_chunk = temp_chunk

            # Add the remaining chunk
            if current_chunk:
                chunks.append(current_chunk.strip())

    # Filter out empty chunks
    return [chunk for chunk in chunks if chunk.strip()]


def generate_audio_for_text(text, sender_id):
    """Generate audio for a given text and return the URL."""
    try:
        # Clean text for TTS (remove excessive formatting but keep basic structure)
        clean_text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with space for TTS
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Normalize whitespace

        synthesis_input = texttospeech.SynthesisInput(text=clean_text)
        voice_config = texttospeech.VoiceSelectionParams(
            language_code="fr-FR", name="fr-FR-Wavenet-B"
        )
        audio_config_tts = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response_tts = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice_config, audio_config=audio_config_tts
        )

        safe_sender_id_part = "".join(c if c.isalnum() else "_" for c in sender_id)
        audio_filename = f"output_{safe_sender_id_part}_{uuid.uuid4()}.mp3"
        audio_filepath = os.path.join(STATIC_DIR_PATH, audio_filename)

        with open(audio_filepath, "wb") as out_file:
            out_file.write(response_tts.audio_content)

        return f"/{STATIC_DIR_NAME}/{audio_filename}"
    except Exception as e:
        print(f"Error during Google TTS for '{sender_id}': {e}")
        return None


@app.route('/process_input', methods=['POST'])
def process_input_route():
    sender_id = request.form.get('sender', f"user_{uuid.uuid4()}")
    audio_file = request.files.get('audio_data')
    text_message_from_form = request.form.get('message')

    user_text_for_rasa = ""

    if audio_file:
        print(f"Processing audio for sender '{sender_id}'...")
        try:
            audio_content = audio_file.read()
            audio_input = speech.RecognitionAudio(content=audio_content)
            config_stt = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                sample_rate_hertz=48000,
                language_code="fr-FR",
                enable_automatic_punctuation=True,
            )
            response_stt = stt_client.recognize(config=config_stt, audio=audio_input)
            if response_stt.results:
                user_text_for_rasa = response_stt.results[0].alternatives[0].transcript
                print(f"Google STT Transcribed Text for '{sender_id}': {user_text_for_rasa}")
            else:
                print(f"Google STT: No transcription results for '{sender_id}'.")
                user_text_for_rasa = " "
        except Exception as e:
            print(f"Error during Google STT for '{sender_id}': {e}")
            return jsonify({
                "error_stt": f"STT processing error: {e}",
                "messages": [{
                    "text": "Désolé, je n'ai pas pu traiter votre audio.",
                    "audio_url": None
                }],
                "user_input_text": ""
            }), 500
    elif text_message_from_form:
        print(f"Processing text message for sender '{sender_id}': {text_message_from_form}")
        user_text_for_rasa = text_message_from_form
    else:
        print("Error: No audio data or text message received in the request.")
        return jsonify({"error": "No audio data or text message received"}), 400

    # --- Send to Rasa ---
    bot_reply_text_from_rasa = "Désolé, une erreur de communication est survenue avec l'assistant."
    try:
        print(f"Sending to Rasa for '{sender_id}': '{user_text_for_rasa}'")
        rasa_payload = {"sender": sender_id, "message": user_text_for_rasa}
        rasa_response = requests.post(RASA_SERVER_URL, json=rasa_payload, timeout=240)
        rasa_response.raise_for_status()
        rasa_data = rasa_response.json()
        if rasa_data and len(rasa_data) > 0:
            bot_reply_text_from_rasa = " ".join(
                [item.get('text', '') for item in rasa_data if item.get('text')]).strip()
        if not bot_reply_text_from_rasa:
            bot_reply_text_from_rasa = "Je n'ai pas de réponse spécifique à cela. Pouvez-vous reformuler ?"
        print(f"Rasa response text for '{sender_id}': {bot_reply_text_from_rasa}")
    except requests.exceptions.Timeout:
        print(f"Error calling Rasa for '{sender_id}': Request timed out")
    except requests.exceptions.RequestException as e:
        print(f"Error calling Rasa for '{sender_id}': {e}")
    except Exception as e:
        print(f"Error processing Rasa response for '{sender_id}': {e}")

    # --- Split the response while preserving formatting ---
    text_chunks = split_text_preserving_format(bot_reply_text_from_rasa, max_length=200)

    # --- Generate audio and prepare response for each chunk ---
    messages = []
    for i, chunk in enumerate(text_chunks):
        audio_url = generate_audio_for_text(chunk, f"{sender_id}_{i}")
        messages.append({
            "text": chunk,
            "audio_url": audio_url
        })

    return jsonify({
        "user_input_text": user_text_for_rasa,
        "messages": messages
    })


@app.route(f'/{STATIC_DIR_NAME}/<path:filename>')
def serve_static_audio(filename):
    print(f"Serving static audio file: {filename}")
    return send_from_directory(STATIC_DIR_PATH, filename, as_attachment=False)


if __name__ == '__main__':
    print(f"Flask backend server starting...")
    app.run(host='0.0.0.0', port=5001, debug=True)