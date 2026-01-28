import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import gc
import shutil
import tempfile
import traceback

import streamlit as st

from pydub import AudioSegment
from huggingface_hub import login

# -------------------------
# Helpers: streamlit caching compatibility
# -------------------------
try:
    cache_resource = st.cache_resource
    cache_data = st.cache_data
except AttributeError:
    cache_resource = st.experimental_singleton
    cache_data = st.experimental_memo

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

# Simple environment diagnostics (helps when host OOMs or is killed)
def diagnostics():
    import sys
    import platform
    st.write("Python:", sys.version.splitlines()[0])
    st.write("Streamlit:", st.__version__)
    st.write("Platform:", platform.platform())
    try:
        import psutil
        mem = psutil.virtual_memory()
        st.write(f"RAM: {mem.total/1e9:.2f} GB total, {mem.available/1e9:.2f} GB available")
    except Exception:
        # psutil may not be available on the host
        pass

# Don't proceed until we have both upload and hf token
if not uploaded_file or not hf_token:
    st.info("Provide a Hugging Face token and upload an audio file to begin.")
    diagnostics()
    st.stop()

# -------------------------
# Lazy & cached pipeline loader
# -------------------------
@cache_resource
def get_pipeline(hf_token: str):
    """
    Load the pyannote Pipeline lazily and cache it for the session.
    We set HUGGINGFACE_HUB_TOKEN so the HF client can authenticate.
    """
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    # Import here to avoid heavy import at module load time
    from pyannote.audio import Pipeline

    # Use the pretrained pyannote diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    # Some Pipeline objects may support .to(device) — call only if present
    try:
        if hasattr(pipeline, "to"):
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            try:
                pipeline.to(device)
            except Exception:
                # Moving to device may not be supported; ignore and proceed on CPU
                pass
    except Exception:
        # Defensive: any unexpected issue shouldn't crash the import path
        pass

    return pipeline

# -------------------------------------------------
# Run separation (triggered by button)
# -------------------------------------------------
if st.button("Separate Speakers", type="primary"):

    progress = st.progress(0)
    status = st.empty()

    try:
        # Ensure HF auth is available for pyannote
        login(token=hf_token)

        # Save upload to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as f:
            f.write(uploaded_file.read())
            input_path = f.name

        progress.progress(10)
        status.text("Converting audio to mono WAV (16kHz)...")

        # Convert to mono WAV (16000 Hz) for diarization
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wav_path = f.name
            audio.export(wav_path, format="wav")

        # free memory for large models
        del audio
        gc.collect()

        progress.progress(30)
        status.text("Loading diarization pipeline (cached)...")

        # Load (cached) pipeline — this is the heavy step
        pipeline = get_pipeline(hf_token)

        progress.progress(60)
        status.text("Running diarization...")

        # Run diarization
        ann = pipeline(wav_path)

        progress.progress(75)
        status.text("Splitting audio by speaker...")

        # Load full (mono) wav for slicing
        audio_full = AudioSegment.from_wav(wav_path)
        duration_ms = len(audio_full)

        speaker_audio = {}

        # ann.itertracks(yield_label=True) yields (segment, track, label) in pyannote
        for segment, _, speaker in ann.itertracks(yield_label=True):
            if speaker not in speaker_audio:
                speaker_audio[speaker] = AudioSegment.silent(duration=duration_ms)

            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)

            speaker_audio[speaker] = speaker_audio[speaker].overlay(
                audio_full[start_ms:end_ms],
                position=start_ms,
            )

        speakers = list(speaker_audio.keys())
        if len(speakers) < 2:
            st.error("Only one speaker detected.")
            raise SystemExit

        # Export the first two speakers
        temp_paths = []
        for i in (0, 1):
            sp_path = None
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as fout:
                sp_path = fout.name
                speaker_audio[speakers[i]].export(sp_path, format=output_format)
                temp_paths.append(sp_path)

        # Store bytes in session state for downloads
        with open(temp_paths[0], "rb") as b:
            st.session_state.speaker1_bytes = b.read()
        with open(temp_paths[1], "rb") as b:
            st.session_state.speaker2_bytes = b.read()
        st.session_state.output_format = output_format

        # Cleanup temp files
        for p in [input_path, wav_path] + temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

        progress.progress(100)
        status.text("Done")

    except Exception as e:
        # Show full traceback in Streamlit UI for debugging
        st.error("Separation failed — see traceback below.")
        st.exception(e)
        st.text(traceback.format_exc())
        # Also print diagnostics to help detect OOM/timeouts on host
        diagnostics()
        raise

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
