import os
import re
import gdown
import requests
import whisper
from pathlib import Path
from tqdm import tqdm
from utils import *
import argparse
import warnings
warnings.filterwarnings("ignore")



def transcribe_video(video_path, output_path=None, model_size="base", language="hi"):
    """
    Transcribe a video file using OpenAI's Whisper model.
    
    Args:
        video_path (str): Path to the video file to transcribe
        output_path (str, optional): Path to save the transcript. If None, uses the video filename with .txt extension
        model_size (str, optional): Whisper model size: "tiny", "base", "small", "medium", or "large"
        language (str, optional): Language hint for the transcription model (e.g., "hi" for Hindi, "en" for English)
        
    Returns:
        str: Path to the saved transcript file
    """
    if not os.path.exists(video_path):
        log_text('logs/question1.txt',FileNotFoundError(f"Video file not found: {video_path}"))
    
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + ".txt"
    
    log_text('logs/question1.txt',f"Loading Whisper {model_size} model...")
    model = whisper.load_model(model_size)
    
    log_text('logs/question1.txt',f"Transcribing {os.path.basename(video_path)}...")
    with tqdm(total=100, desc="Transcribing", bar_format='{l_bar}{bar}| {elapsed}') as pbar:
        result = model.transcribe(
            video_path, 
            task="transcribe",
            language=language, 
            verbose=False       
        )
        pbar.update(100)  
    
    return result['text']




def preprocess_transcript(text, file_path):
    """
    Removes filler words like "um", "uh", etc.

    Args:
        text (str): Text string containing the transcript generated by model

    Returns:
        cleaned_text (str): Text string after removing all the filler words 
    """
    filler_words = [r"\bum\b", r"\buh\b", r"\blike\b", r"\buhm\b", r"\buhhmm\b", r"\ba\b", r"\bhmm\b"]
    pattern = re.compile("|".join(filler_words), flags=re.IGNORECASE)

    cleaned_text = pattern.sub("", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    with open(file_path+'cleaned_transcript.txt', 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    log_text('logs/question1.txt',f"Transcript cleaned and saved to: {file_path}cleaned_transcript.txt")

    return file_path
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Download and transcribe Google Drive videos")
    parser.add_argument("--video_path", help="Location of the MP4 video file")
    parser.add_argument("--target-dir", default="data", help="Directory to save the downloaded video")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],  help="Whisper model size to use for transcription")
    
    args = parser.parse_args()

    if args.video_path:
        
        transcript = transcribe_video(args.video_path, model_size=args.model)
        cleaned_transcript_path = preprocess_transcript(transcript, 'data/transcripts/')

        log_text('logs/question1.txt',f"Process complete.")


