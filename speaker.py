import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import torch
import tempfile
import os
import gc
import shutil

from pyannote.audio import Pipeline
from pydub import AudioSegment
from huggingface_hub import login

# -------------------------------------------------
# Startup: FFmpeg validation
# -------------------------------------------------
def check_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        st.error(
            """
            ❌ **FFmpeg not found**

            This app requires **ffmpeg** and **ffprobe** in PATH.

            **Windows fix**
            1. Download from https://www.gyan.dev/ffmpeg/builds/
            2. Extract to `C:\\ffmpeg`
            3. Add `C:\\ffmpeg\\bin` to PATH
            4. Restart Streamlit
            """
        )
        st.stop()

check_ffmpeg()

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Speaker Separation", layout="centered")
st.title("Speaker Separation")
st.write("Separate two speakers into individual mono audio files.")

# -------------------------------------------------
# Session state
# -------------------------------------------------
if "speaker1_bytes" not in st.session_state:
    st.session_state.speaker1_bytes = None
    st.session_state.speaker2_bytes = None
    st.session_state.output_format = None

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    hf_token = st.text_input("Hugging Face Token", type="password")
    output_format = st.selectbox("Output format", ["wav", "m4a"])

# -------------------------------------------------
# Upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "m4a", "flac", "ogg"]
)

if not uploaded_file or not hf_token:
    st.stop()

# -------------------------------------------------
# Run separation
# -------------------------------------------------
if st.button("Separate Speakers", type="primary"):

    try:
        login(token=hf_token)

        progress = st.progress(0)
        status = st.empty()

        # Save upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as f:
            f.write(uploaded_file.read())
            input_path = f.name

        progress.progress(20)

        # Convert to mono WAV
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wav_path = f.name
            audio.export(wav_path, format="wav")

        del audio
        gc.collect()

        progress.progress(40)

        # Load diarization pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        )
        pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # 🔴 FIXED API: pipeline returns Annotation directly
        ann = pipeline(wav_path)

        progress.progress(60)

        # Separate speakers
        audio_full = AudioSegment.from_wav(wav_path)
        duration_ms = len(audio_full)

        speaker_audio = {}

        for segment, _, speaker in ann.itertracks(yield_label=True):
            if speaker not in speaker_audio:
                speaker_audio[speaker] = AudioSegment.silent(duration=duration_ms)

            speaker_audio[speaker] = speaker_audio[speaker].overlay(
                audio_full[int(segment.start * 1000):int(segment.end * 1000)],
                position=int(segment.start * 1000)
            )

        speakers = list(speaker_audio.keys())
        if len(speakers) < 2:
            st.error("Only one speaker detected.")
            st.stop()

        # Export to temp + store in session
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as f1:
            speaker_audio[speakers[0]].export(f1.name, format=output_format)
            with open(f1.name, "rb") as b:
                st.session_state.speaker1_bytes = b.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as f2:
            speaker_audio[speakers[1]].export(f2.name, format=output_format)
            with open(f2.name, "rb") as b:
                st.session_state.speaker2_bytes = b.read()

        st.session_state.output_format = output_format

        # Cleanup files
        for p in [input_path, wav_path, f1.name, f2.name]:
            try:
                os.remove(p)
            except:
                pass

        progress.progress(100)
        status.text("Done")

    except Exception as e:
        st.error(str(e))
        st.stop()

# -------------------------------------------------
# Downloads (safe across reruns)
# -------------------------------------------------
if st.session_state.speaker1_bytes:

    st.success("Separation complete")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "Download Speaker 1",
            st.session_state.speaker1_bytes,
            file_name=f"speaker1.{st.session_state.output_format}",
            mime=f"audio/{st.session_state.output_format}",
        )

    with col2:
        st.download_button(
            "Download Speaker 2",
            st.session_state.speaker2_bytes,
            file_name=f"speaker2.{st.session_state.output_format}",
            mime=f"audio/{st.session_state.output_format}",
        )
