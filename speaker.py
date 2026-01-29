import os
import gc
import shutil
import tempfile
import traceback
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
from pydub import AudioSegment
from huggingface_hub import login

# -------------------------------------------------
# Streamlit cache compatibility
# -------------------------------------------------
try:
    cache_resource = st.cache_resource
except AttributeError:
    cache_resource = st.experimental_singleton

# -------------------------------------------------
# Hard limits (production safety)
# -------------------------------------------------
MAX_FILE_MB = 50
MAX_DURATION_SECONDS = 30 * 60  # 30 minutes

# -------------------------------------------------
# FFmpeg validation
# -------------------------------------------------
def require_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        st.error("FFmpeg and ffprobe must be installed and in PATH.")
        st.stop()

require_ffmpeg()

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
# Session state
# -------------------------------------------------
for key in ("spk1", "spk2", "fmt"):
    st.session_state.setdefault(key, None)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    hf_token = st.text_input("Hugging Face Token", type="password")
    output_format = st.selectbox("Output format", ("wav", "m4a"))

# -------------------------------------------------
# Upload
# -------------------------------------------------
uploaded = st.file_uploader(
    "Upload audio file",
    type=("wav", "mp3", "m4a", "flac", "ogg")
)

if not uploaded or not hf_token:
    st.info("Provide a Hugging Face token and upload an audio file.")
    st.stop()

# -------------------------------------------------
# Validate upload
# -------------------------------------------------
file_size_mb = len(uploaded.getbuffer()) / (1024 * 1024)
if file_size_mb > MAX_FILE_MB:
    st.error(f"File too large ({file_size_mb:.1f} MB). Limit is {MAX_FILE_MB} MB.")
    st.stop()

# -------------------------------------------------
# Authenticate once per session
# -------------------------------------------------
if "hf_authed" not in st.session_state:
    login(token=hf_token)
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    st.session_state.hf_authed = True

# -------------------------------------------------
# Load diarization pipeline (cached, token-free)
# -------------------------------------------------
@cache_resource
def load_pipeline():
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    try:
        import torch
        if torch.cuda.is_available():
            pipeline.to("cuda")
    except Exception:
        pass

    return pipeline

# -------------------------------------------------
# Separation
# -------------------------------------------------
if st.button("Separate Speakers", type="primary"):

    progress = st.progress(0)
    status = st.empty()

    tmp_files = []

    try:
        # Save upload
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(uploaded.read())
            src_path = f.name
            tmp_files.append(src_path)

        progress.progress(10)
        status.text("Normalizing audio...")

        audio = AudioSegment.from_file(src_path)
        duration_sec = len(audio) / 1000

        if duration_sec > MAX_DURATION_SECONDS:
            raise RuntimeError("Audio too long for processing.")

        audio = audio.set_channels(1).set_frame_rate(16000)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wav_path = f.name
            audio.export(wav_path, format="wav")
            tmp_files.append(wav_path)

        del audio
        gc.collect()

        progress.progress(40)
        status.text("Loading model...")

        pipeline = load_pipeline()

        progress.progress(60)
        status.text("Running diarization...")

        diarization = pipeline(wav_path)

        progress.progress(75)
        status.text("Reconstructing speakers...")

        full_audio = AudioSegment.from_wav(wav_path)

        segments = {}
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.setdefault(speaker, []).append(
                full_audio[int(segment.start * 1000):int(segment.end * 1000)]
            )

        if len(segments) < 2:
            raise RuntimeError("Less than two speakers detected.")

        # Deterministic ordering by total speech duration
        ordered = sorted(
            segments.items(),
            key=lambda x: sum(len(s) for s in x[1]),
            reverse=True
        )[:2]

        outputs = []
        for speaker, chunks in ordered:
            combined = AudioSegment.empty()
            for c in chunks:
                combined += c
            outputs.append(combined)

        for i, audio_out in enumerate(outputs):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as f:
                audio_out.export(f.name, format=output_format)
                tmp_files.append(f.name)
                with open(f.name, "rb") as b:
                    if i == 0:
                        st.session_state.spk1 = b.read()
                    else:
                        st.session_state.spk2 = b.read()

        st.session_state.fmt = output_format

        progress.progress(100)
        status.text("Completed")

    except Exception as e:
        st.error("Processing failed.")
        st.exception(e)
        st.text(traceback.format_exc())

    finally:
        for p in tmp_files:
            try:
                os.remove(p)
            except Exception:
                pass
        gc.collect()

# -------------------------------------------------
# Downloads
# -------------------------------------------------
if st.session_state.spk1:
    st.success("Separation complete")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download Speaker 1",
            st.session_state.spk1,
            file_name=f"speaker1.{st.session_state.fmt}",
            mime=f"audio/{st.session_state.fmt}",
        )

    with c2:
        st.download_button(
            "Download Speaker 2",
            st.session_state.spk2,
            file_name=f"speaker2.{st.session_state.fmt}",
            mime=f"audio/{st.session_state.fmt}",
        )
