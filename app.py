import streamlit as st
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx.all import crop
import whisper
import os
import uuid

st.set_page_config(page_title="AttentionX AI", layout="centered")

st.title("🎬 AttentionX AI")
st.markdown("### Convert long videos into viral clips using AI 🚀")

uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4"])

if uploaded_file is not None:
    try:
        with st.spinner("⏳ Processing your video..."):

            st.video(uploaded_file)

            # 🔑 Unique filenames (no overwrite)
            uid = str(uuid.uuid4())
            video_path = f"input_{uid}.mp4"
            audio_path = f"audio_{uid}.wav"

            # Save video
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

            # 🔊 AUDIO EXTRACTION
            st.write("🔊 Extracting audio...")
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path)

            st.success("✅ Audio Extracted!")

            # 🤖 SPEECH TO TEXT
            st.write("🤖 Converting speech to text...")
            model = whisper.load_model("tiny")
            result = model.transcribe(audio_path, fp16=False)

            st.success("✅ Transcription Done!")

            st.subheader("📝 Transcript")
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

                clip = video.subclip(start, end)

                # 📱 Convert to vertical (Reels format)
                clip = crop(
                    clip,
                    width=int(clip.h * 9 / 16),
                    height=clip.h,
                    x_center=clip.w / 2,
                    y_center=clip.h / 2
                )

                # 🎯 Add captions (safe fallback if ImageMagick not installed)
                try:
                    txt = TextClip(
                        segment["text"],
                        fontsize=40,
                        color='yellow',
                        bg_color='black',
                        method='caption',
                        size=(clip.w, None)
                    )

                    txt = txt.set_position(("center", "bottom")).set_duration(clip.duration)
                    final = CompositeVideoClip([clip, txt])

                except Exception:
                    final = clip  # fallback if captions fail

                filename = f"outputs/final_{uid}_{i}.mp4"
                final.write_videofile(filename, codec="libx264", audio_codec="aac")

                st.success(f"🔥 Clip {i+1} Generated!")
                st.video(filename)

                # 📥 Download button
                with open(filename, "rb") as file:
                    st.download_button(
                        label=f"📥 Download Clip {i+1}",
                        data=file,
                        file_name=f"clip_{i+1}.mp4",
                        mime="video/mp4"
                    )

            video.close()

            st.success("🎉 All clips generated successfully!")

    except Exception as e:
        st.error(f"❌ Error: {e}")
