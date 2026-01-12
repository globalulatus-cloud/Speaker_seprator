import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import torch
import tempfile
import os
import gc

from pyannote.audio import Pipeline
from pydub import AudioSegment
from huggingface_hub import login

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Speaker Separation",
    layout="centered"
)

st.title("Speaker Separation")
st.write("Separate two speakers into individual mono audio files.")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("Settings")

    hf_token = st.text_input(
        "Hugging Face Token",
        type="password",
        help="Create a READ token at https://huggingface.co/settings/tokens"
    )

    output_format = st.selectbox(
        "Output format",
        ["wav", "m4a"]
    )

    st.markdown("---")
    st.markdown(
        """
        **Instructions**
        1. Accept model terms:
           - https://huggingface.co/pyannote/speaker-diarization
           - https://huggingface.co/pyannote/segmentation
        2. Paste HF token
        3. Upload audio
        4. Click Separate Speakers
        """
    )

# -------------------------------------------------
# Upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "m4a", "flac", "ogg"]
)

if not uploaded_file:
    st.info("Upload an audio file to begin.")
    st.stop()

if not hf_token:
    st.warning("Please enter your Hugging Face token.")
    st.stop()

# -------------------------------------------------
# Action
# -------------------------------------------------
if st.button("Separate Speakers", type="primary"):

    try:
        # Authenticate HF (global auth, no kwargs)
        login(token=hf_token)

        progress = st.progress(0)
        status = st.empty()

        # -------------------------------------------------
        # Save uploaded file
        # -------------------------------------------------
        status.text("Loading audio...")
        progress.progress(10)

        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded_file.read())
            input_path = f.name

        # -------------------------------------------------
        # Convert to mono WAV
        # -------------------------------------------------
        status.text("Converting audio...")
        progress.progress(25)

        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wav_path = f.name
            audio.export(wav_path, format="wav")

        del audio
        gc.collect()

        # -------------------------------------------------
        # Load model
        # -------------------------------------------------
        status.text("Loading diarization model...")
        progress.progress(40)

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)

        # -------------------------------------------------
        # Run diarization
        # -------------------------------------------------
        status.text("Running speaker diarization...")
        progress.progress(60)

        diarization = pipeline(wav_path)
        ann = diarization.speaker_diarization

        # -------------------------------------------------
        # Separate speakers
        # -------------------------------------------------
        status.text("Separating speakers...")
        progress.progress(80)

        audio_full = AudioSegment.from_wav(wav_path)
        duration_ms = len(audio_full)

        speaker_audio = {}

        for segment, _, speaker in ann.itertracks(yield_label=True):
            if speaker not in speaker_audio:
                speaker_audio[speaker] = AudioSegment.silent(duration=duration_ms)

            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)

            speaker_audio[speaker] = speaker_audio[speaker].overlay(
                audio_full[start_ms:end_ms],
                position=start_ms
            )

        speakers = list(speaker_audio.keys())

        if len(speakers) < 2:
            st.warning("Only one speaker detected. Cannot separate.")
            st.stop()

        # -------------------------------------------------
        # Export to temp files
        # -------------------------------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as f1:
            spk1_path = f1.name
            speaker_audio[speakers[0]].export(spk1_path, format=output_format)

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as f2:
            spk2_path = f2.name
            speaker_audio[speakers[1]].export(spk2_path, format=output_format)

        # -------------------------------------------------
        # READ FILES INTO MEMORY (CRITICAL FIX)
        # -------------------------------------------------
        with open(spk1_path, "rb") as f:
            speaker1_bytes = f.read()

        with open(spk2_path, "rb") as f:
            speaker2_bytes = f.read()

        progress.progress(100)
        status.text("Done!")

        st.success("Separation complete")

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "Download Speaker 1",
                data=speaker1_bytes,
                file_name=f"speaker1.{output_format}",
                mime=f"audio/{output_format}"
            )

        with col2:
            st.download_button(
                "Download Speaker 2",
                data=speaker2_bytes,
                file_name=f"speaker2.{output_format}",
                mime=f"audio/{output_format}"
            )

        # -------------------------------------------------
        # Cleanup (SAFE: after bytes are loaded)
        # -------------------------------------------------
        for p in [input_path, wav_path, spk1_path, spk2_path]:
            try:
                os.remove(p)
            except:
                pass

    except Exception as e:
        st.error(f"Error: {e}")
