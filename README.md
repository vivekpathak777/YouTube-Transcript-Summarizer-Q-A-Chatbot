# 🎬 YouTube Transcript Summarizer + Q&A Chatbot

An interactive **Streamlit app** that fetches transcripts from YouTube videos, generates concise **bullet-point summaries** using **Google Gemini**, and lets you **ask grounded questions** about the video content.

---

## ✨ Features
- **Robust transcript fetching**  
  - First tries [`youtube-transcript-api`](https://pypi.org/project/youtube-transcript-api/)  
  - Falls back silently to [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) if needed  
- **Flexible input**: works with standard YouTube links, `youtu.be`, `/shorts/`, `/embed/`, or raw 11-char IDs  
- **Bullet-only summaries**  
  - 2–3 sentences per bullet  
  - ≤10 bullets total  
  - Configurable word count target (100–1000 words)  
- **Grounded Q&A**  
  - Ask questions directly about the transcript  
  - Answers cite transcript content, no hallucinations  
- **Download support**  
  - Save summaries and transcripts as `.txt` files  
- **Streamlit UI** with persistent chat history  

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/youtube-transcript-summarizer.git
cd youtube-transcript-summarizer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

If you don’t have `yt-dlp` installed globally, the app auto-installs it at runtime.

### 3. Run the app
```bash
streamlit run app.py
```

You’ll see a local URL (and a public one if you’re using Ngrok in Colab).

---

## 🛠 Requirements
- Python 3.9+  
- [Streamlit](https://streamlit.io)  
- [youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/)  
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) (for fallback subtitle fetch)  
- [google-generativeai](https://ai.google.dev/) (Gemini client)  
- [pyngrok](https://pyngrok.readthedocs.io/en/latest/) (optional, for Colab tunneling)  

---

## 🔑 Gemini API Key
1. Get an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).  
2. Enter the key in the app’s sidebar when prompted.  

---

## 📦 Usage
1. Paste a YouTube video URL or ID.  
2. Choose:
   - **Summary style**: Short or Long  
   - **Word target**: 100–1000 words  
3. Click **Generate Summary**.  
4. Ask questions in the **Q&A section** to get grounded answers.  
5. Download the transcript or summary if needed.  

---

## 🖼 Screenshots

*(Add your own screenshots here — e.g., summary view, Q&A chat, downloads)*

---

## ⚠️ Notes
- Some videos may not have transcripts (private, region-locked, members-only, or disabled by the uploader).  
- Long videos are automatically **chunked** to fit Gemini’s context window.  
- Fallback to `yt-dlp` is silent — you’ll just see the summary working.  

---

## 📜 License
MIT License. See [LICENSE](LICENSE) for details.
