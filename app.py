from flask import Flask, render_template, request
import openai
import whisper
import os
model = whisper.load_model("base")
print("Current Working Directory:", os.getcwd())
# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("meeting.mp3")
print(f"Audio duration: seconds")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)

app = Flask(__name__)
openai.api_key = OPENAI_API_KEY

def transcribe_audio(file_path):
    audio_file = open(file_path, 'rb')
    response = openai.Audio.transcriptions.create(
        file=audio_file,
        model="whisper-1"
    )
    return response['text']

def summarize_text(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a proficient AI specializing in summarizing meeting minutes."},
            {"role": "user", "content": transcription}
        ]
    )
    return response.choices[0].message['content']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    file_path = f"/tmp/{file.filename}"
    file.save(file_path)

    transcription = transcribe_audio(file_path)
    summary = summarize_text(transcription)

    return summary

if __name__ == '__main__':
    app.run(debug=True)
