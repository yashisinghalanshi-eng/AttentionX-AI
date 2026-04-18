import streamlit as st
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx import Crop
import whisper
import os

st.set_page_config(page_title="AttentionX AI", layout="centered")

st.title("🎬 AttentionX AI - Video to Viral Clips")

uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4"])

if uploaded_file is not None:

    st.video(uploaded_file)

    # Save video
    with open("input.mp4", "wb") as f:
        f.write(uploaded_file.read())

    # 🔊 AUDIO EXTRACTION
    st.write("🔊 Extracting audio...")
    video = VideoFileClip("input.mp4")
    video.audio.write_audiofile("audio.wav")

    st.success("✅ Audio Extracted!")

    # 🤖 SPEECH TO TEXT
    st.write("🤖 Converting speech to text...")
    model = whisper.load_model("tiny")   # fast model
    result = model.transcribe("audio.wav", fp16=False)

    st.success("✅ Transcription Done!")

    st.write("📝 Transcript:")
    st.write(result["text"])

    # 🔍 IMPORTANT MOMENTS DETECTION
    st.write("🔍 Finding important moments...")

    keywords = ["important", "success", "mistake", "never", "learn"]
    important_segments = []

    for segment in result["segments"]:
        text = segment["text"].lower()
        if any(word in text for word in keywords):
            important_segments.append(segment)

    # fallback
    if len(important_segments) == 0:
        important_segments = result["segments"][:3]

    st.success("✅ Important moments detected!")

    st.write("🎬 Generating viral clips...")

    os.makedirs("outputs", exist_ok=True)

    # ✂️ CREATE MULTIPLE CLIPS
    for i, segment in enumerate(important_segments[:3]):

        start = segment["start"]
        end = segment["end"]

        clip = video[start:end]

        # 📱 Convert to vertical (Reels format)
        clip = Crop(width=int(clip.h * 9 / 16), x_center=clip.w / 2).apply(clip)

        # 🎯 Add captions
        txt = TextClip(
            text=segment["text"],
            font_size=40,
            color='yellow',
            bg_color='black'
        )

        txt = txt.with_position('bottom').with_duration(clip.duration)

        final = CompositeVideoClip([clip, txt])

        filename = f"outputs/final_{i}.mp4"
        final.write_videofile(filename)

        st.success(f"🔥 Clip {i+1} Generated!")
        st.video(filename)

    st.success("🎉 All clips generated successfully!")





