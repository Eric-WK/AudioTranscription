import streamlit as st 
from tempfile import NamedTemporaryFile
from audio_recorder_streamlit import audio_recorder
import whisper
from model import AudioTranscriber
## create a title 
st.title("Audio Transcriber")

## create a subheader
st.subheader("Upload an audio file to transcribe it")

## create a file uploader
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

## or record your own audio 
st.subheader("Press the button to record your own audio ")
audio_bytes = audio_recorder(pause_threshold=5.0)
if audio_bytes:
    ## show the audio
    st.audio(audio_bytes, format="audio/wav")

## create a button to transcribe the audio file
if st.button("Transcribe"):
    AUDIO = audio_file or audio_bytes
    if audio_file or audio_bytes:
        with NamedTemporaryFile(suffix="mp3") as MY_AUDIO:
            MY_NEW_AUDIO = AUDIO.getvalue() if type(AUDIO) != bytes else AUDIO
            MY_AUDIO.write(MY_NEW_AUDIO)
            MY_AUDIO.seek(0)
            ## create an instance of the AudioTranscriber class
            transcriber = AudioTranscriber("base", MY_AUDIO.name)
            ## detect the language 
            det_lang = transcriber.mel_spec_audio_detection()["language"]
            st.write(f"Detected Language: {det_lang}")
            ## set the options 
            options = whisper.DecodingOptions(fp16=False if transcriber.model.device.type == 'cpu' else True)
            ## transcribe the audio file without trimming it 
            transcription_no_trim = transcriber.transcribe_audio_no_trim()['text']
            ## write the transcription to the screen
            st.write(f"Transcription: \n {transcription_no_trim.strip()}")