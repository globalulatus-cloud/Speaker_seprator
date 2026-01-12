# Speaker Separation App (PyAnnote + Streamlit)

This project provides a **local Streamlit web application** for **automatic speaker separation** using **PyAnnote AI**.

It separates a 2-speaker audio file into two mono audio files:
- `speaker1.wav`
- `speaker2.wav`

Each output contains only one speaker, with silence elsewhere.

---

## ⚠️ Important Constraints (Read First)

This project **ONLY works** with:

- **Python 3.10**
- **Pinned dependency versions**
- **Local execution (Windows / macOS / Linux)**

It will **NOT work reliably** on:
- Python 3.11+
- Google Colab (unless Python 3.10 is available)
- Replit
- Lovable backend (frontend-only possible)

---

## 🧱 Tech Stack

- Python 3.10
- Streamlit
- PyAnnote Audio 3.1.1
- Torch / Torchaudio
- Hugging Face Hub
- FFmpeg (for audio processing)
