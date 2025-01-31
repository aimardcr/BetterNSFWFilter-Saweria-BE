from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
from transformers import pipeline
from transformers.pipelines import PipelineException
from PIL import Image
from cachetools import Cache
import tensorflow as tf
import yt_dlp
import cv2
import tempfile
from urllib.parse import urlparse
import re
import hashlib
import requests
import io

nsfw_cache = Cache(maxsize=1000)
video_cache = Cache(maxsize=100)

model = pipeline("image-classification", model="falconsai/nsfw_image_detection")

app = Flask(__name__)
app.config['REELS_FOLDER'] = 'static/video/reels'

logging.disable(logging.WARNING) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.makedirs(app.config['REELS_FOLDER'], exist_ok=True)

CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "stream-key"]
    }
})

YDL_OPTS = {
    'format': 'best[ext=mp4]',
    'quiet': True,
    'no_warnings': True,
}

def sha256(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()

def get_video_hash(video_path):
    sha256_hash = hashlib.sha256()
    with open(video_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def get_platform(url):
    domain = urlparse(url).netloc.lower()
    path = urlparse(url).path
    if 'youtube' in domain or 'youtu.be' in domain:
        return 'youtube'
    elif 'tiktok' in domain:
        return 'tiktok'
    elif 'instagram' in domain and '/reel/' in path:
        return 'instagram'
    else:
        return None

def extract_instagram_id(url):
    try:
        regex = r'(?:https?:\/\/)?(?:www\.)?instagram\.com(?:\/reels\/|\/reel\/)([^\/?]+)(?:\S+)?'
        match = re.search(regex, url, re.IGNORECASE)
        
        if match:
            return match.group(1)
        
        return None
    except Exception:
        return None

def download_video(url):
    try:
        platform = get_platform(url)

        if platform == 'instagram':
            reel_id = extract_instagram_id(url)
            if not reel_id:
                return None, "Could not extract Instagram reel ID"
            
            output_path = os.path.join(app.config['REELS_FOLDER'], f'{reel_id}.mp4')
            
            if os.path.exists(output_path):
                return output_path, None
            
            opts = YDL_OPTS.copy()
            opts['outtmpl'] = output_path

            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            
            if os.path.exists(output_path):
                return output_path, None
            return None, "Failed to download Instagram reel"

        temp_dir = tempfile.mkdtemp()
        
        opts = YDL_OPTS.copy()
        opts['outtmpl'] = os.path.join(temp_dir, 'video.mp4')
        
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        
        video_path = os.path.join(temp_dir, 'video.mp4')
        if os.path.exists(video_path):
            return video_path, None
            
        return None, "Failed to download video"
    except Exception as e:
        return None, str(e)

def extract_frames(video_path, sample_rate=1):
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / sample_rate)
        
        frame_count = 0
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                pil_image = Image.fromarray(frame)
                frames.append(pil_image)
            
            frame_count += 1
            
        cap.release()
        return frames, None
    except Exception as e:
        return None, str(e)

def get_image_hash(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return hashlib.sha256(img_byte_arr).hexdigest()

def classify_frame(frame):
    frame_hash = get_image_hash(frame)
    
    cached_result = nsfw_cache.get(frame_hash)
    if cached_result is not None:
        return cached_result
    
    result = model(frame)
    if isinstance(result, list):
        result = result[0]
    
    nsfw_score = 0
    for pred in result:
        if isinstance(pred, dict) and pred.get('label') == 'nsfw':
            nsfw_score = pred.get('score', 0)
            break
    
    nsfw_cache[frame_hash] = nsfw_score
    return nsfw_score

@app.route('/check-video', methods=['POST'])
def check_video():
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400
    
    url = data['url']
    
    url_hash = sha256(url)
    cached_result = video_cache.get(url_hash)
    if cached_result is not None:
        return jsonify(cached_result)
    
    if not is_valid_url(url):
        return jsonify({'error': 'Invalid URL'}), 400
    
    platform = get_platform(url)
    if not platform:
        return jsonify({'error': 'Unsupported platform'}), 400
    
    video_path, error = download_video(url)
    if error:
        return jsonify({'error': f'Failed to download video: {error}'}), 400

    video_hash = get_video_hash(video_path)
    cached_result = video_cache.get(video_hash)
    if cached_result is not None:
        if platform != 'instagram':
            os.remove(video_path)
        return jsonify(cached_result)

    frames, error = extract_frames(video_path)
    if error:
        if platform != 'instagram':
            os.remove(video_path)
        return jsonify({'error': f'Failed to extract frames: {error}'}), 400
    
    nsfw_frames = []
    for i, frame in enumerate(frames):
        nsfw_score = classify_frame(frame)
        if nsfw_score > 0.7:  # 70%
            nsfw_frames.append({
                'frame_number': i,
                'confidence': round(nsfw_score * 100, 1)
            })

    max_confidence = max((frame['confidence'] for frame in nsfw_frames), default=0)

    result = {
        'url': url,
        'platform': platform,
        'is_nsfw': len(nsfw_frames) > 0,
        'max_confidence': max_confidence,
        'nsfw_frames': nsfw_frames
    }

    video_cache[url_hash] = result
    video_cache[video_hash] = result

    if platform != 'instagram':
        os.remove(video_path)

    return jsonify(result)

@app.get('/get-mediashare-template')
def get_mediashare_template():
    stream_key = request.headers.get('stream-key')
    return requests.get('https://backend.saweria.co/template/mediashare', headers={'stream-key': stream_key}).json()

@app.route('/reels/<reel_id>', methods=['GET'])
def get_reel(reel_id):
    video_path = os.path.join(app.config['REELS_FOLDER'], f'{reel_id}.mp4')
    
    if not os.path.exists(video_path):
        return jsonify({'error': 'Reel not found'}), 404
    
    return send_file(video_path, mimetype='video/mp4')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
