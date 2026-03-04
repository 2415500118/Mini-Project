from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import pickle
import numpy as np
import librosa
import os
import tempfile
from pathlib import Path

N_MELS = 128
FIXED_LENGTH = 128
N_FFT = 512
HOP_LENGTH = 256

class SpeakerCNN(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super(SpeakerCNN, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.1)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.4)
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.5)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_layer = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedding_layer(x)
        output = self.classifier(embedding)
        return output, embedding

def _load_and_clean_audio(file_path, sample_rate=16000):
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    if audio is None or len(audio) == 0:
        raise ValueError(f"Empty audio: {file_path}")

    audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
    if len(audio_trimmed) > 0:
        audio = audio_trimmed

    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        audio = audio / peak

    return audio.astype(np.float32), sr

def _compute_log_mel(audio, sr, n_mels=N_MELS):
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    return librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)

def _fixed_crop(log_mel_spec, start_idx, fixed_length=FIXED_LENGTH):
    crop = log_mel_spec[:, start_idx:start_idx + fixed_length]
    if crop.shape[1] < fixed_length:
        pad_width = fixed_length - crop.shape[1]
        crop = np.pad(crop, ((0, 0), (0, pad_width)), mode='constant')
    return crop.astype(np.float32)

def audio_to_melspectrogram(file_path, sample_rate=16000, n_mels=N_MELS, fixed_length=FIXED_LENGTH):
    audio, sr = _load_and_clean_audio(file_path, sample_rate=sample_rate)
    log_mel_spec = _compute_log_mel(audio, sr, n_mels=n_mels)
    if log_mel_spec.shape[1] < fixed_length:
        pad_width = fixed_length - log_mel_spec.shape[1]
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        return log_mel_spec.astype(np.float32)
    return log_mel_spec[:, :fixed_length].astype(np.float32)

def audio_to_melspectrogram_crops(file_path, sample_rate=16000, n_mels=N_MELS, fixed_length=FIXED_LENGTH, num_crops=3):
    audio, sr = _load_and_clean_audio(file_path, sample_rate=sample_rate)
    log_mel_spec = _compute_log_mel(audio, sr, n_mels=n_mels)
    total_frames = log_mel_spec.shape[1]
    if total_frames <= fixed_length or num_crops <= 1:
        return [_fixed_crop(log_mel_spec, 0, fixed_length=fixed_length)]
    max_start = total_frames - fixed_length
    start_positions = np.linspace(0, max_start, num=num_crops, dtype=int)
    return [_fixed_crop(log_mel_spec, int(start), fixed_length=fixed_length) for start in start_positions]

def get_embedding(file_path, model, device):
    spec = audio_to_melspectrogram(file_path)
    spec_tensor = torch.tensor(spec).unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        _, embedding = model(spec_tensor)
    return embedding.squeeze(0).cpu().numpy()

def identify_speaker(audio_path, model, voice_db, device, threshold=0.80):
    from torch.nn.functional import cosine_similarity
    input_embedding = get_embedding(audio_path, model, device)
    input_tensor = torch.tensor(input_embedding).unsqueeze(0)
    best_speaker = None
    best_similarity = -1
    for speaker_name, db_embedding in voice_db.items():
        db_tensor = torch.tensor(db_embedding).unsqueeze(0)
        similarity = cosine_similarity(input_tensor, db_tensor).item()
        if similarity > best_similarity:
            best_similarity = similarity
            best_speaker = speaker_name
    if best_similarity >= threshold:
        status = "Authenticated"
    else:
        status = "Voice Not Found"
        best_speaker = "Unknown"
    return {
        "predicted_name": best_speaker,
        "similarity_score": round(float(best_similarity), 4),
        "status": status
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
voice_db = None
NUM_CLASSES = None

def _safe_torch_load(checkpoint_path, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)

@asynccontextmanager
async def lifespan(app):
    global model, voice_db, NUM_CLASSES
    with open("voice_db.pkl", "rb") as f:
        raw_voice_db = pickle.load(f)
    voice_db = {str(name): np.asarray(emb, dtype=np.float32) for name, emb in raw_voice_db.items()}
    NUM_CLASSES = len(voice_db)
    print(f"[OK] Voice database loaded: {NUM_CLASSES} speakers")
    print(f"  Speakers: {sorted(list(voice_db.keys()))}")
    checkpoint_candidates = ["model_best.pth", "model.pth"]
    checkpoint_path = next((path for path in checkpoint_candidates if os.path.exists(path)), None)
    if checkpoint_path is None:
        raise FileNotFoundError("Neither model_best.pth nor model.pth was found")
    state_dict = _safe_torch_load(checkpoint_path, map_location=device)
    model = SpeakerCNN(num_classes=NUM_CLASSES, embedding_dim=512).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[OK] Model loaded successfully from {checkpoint_path} on {device}")
    yield

app = FastAPI(title="VoxGuard API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "status": "online",
        "device": str(device),
        "speakers": list(voice_db.keys()) if voice_db else [],
        "num_speakers": NUM_CLASSES
    }

@app.get("/speakers")
async def get_speakers():
    if not voice_db:
        raise HTTPException(status_code=500, detail="Database not loaded")
    return {
        "speakers": sorted(list(voice_db.keys())),
        "count": len(voice_db)
    }

@app.post("/authenticate")
async def authenticate_speaker(
    audio: UploadFile = File(...),
    speaker: str = None,
    threshold: float = 0.70
):
    if not speaker:
        raise HTTPException(status_code=400, detail="Speaker name required")
    if speaker not in voice_db:
        raise HTTPException(status_code=400, detail=f"Speaker '{speaker}' not in database")
    if not audio.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.filename).suffix) as tmp_file:
        content = await audio.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    try:
        from torch.nn.functional import cosine_similarity
        input_embedding = get_embedding(tmp_path, model, device)
        input_tensor = torch.tensor(input_embedding).unsqueeze(0)
        claimed_db_embedding = voice_db[speaker]
        claimed_tensor = torch.tensor(claimed_db_embedding).unsqueeze(0)
        similarity = cosine_similarity(input_tensor, claimed_tensor).item()
        all_scores = []
        for db_embedding in voice_db.values():
            db_tensor = torch.tensor(db_embedding).unsqueeze(0)
            all_scores.append(cosine_similarity(input_tensor, db_tensor).item())
        all_scores = sorted(all_scores, reverse=True)
        top1 = all_scores[0] if len(all_scores) > 0 else similarity
        top2 = all_scores[1] if len(all_scores) > 1 else -1.0
        margin = top1 - top2
        effective_threshold = threshold - 0.05 if margin >= 0.08 else threshold
        effective_threshold = max(0.55, effective_threshold)
        authenticated = similarity >= effective_threshold
        return JSONResponse(content={
            "success": True,
            "authenticated": authenticated,
            "similarity": round(float(similarity), 4),
            "claimed_speaker": speaker,
            "threshold": threshold,
            "effective_threshold": round(float(effective_threshold), 4),
            "margin_vs_next": round(float(margin), 4),
            "decision": "AUTHENTICATED ✅" if authenticated else "ACCESS DENIED ❌",
            "confidence": round(max(0, min(100, (similarity - threshold + 0.3) / 0.3 * 100)), 2)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        os.unlink(tmp_path)

@app.post("/verify")
async def verify_speaker(audio: UploadFile = File(...), threshold: float = 0.60):
    if not audio.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    if not voice_db:
        raise HTTPException(status_code=500, detail="Speaker database is empty")
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.filename).suffix) as tmp_file:
        content = await audio.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    try:
        from torch.nn.functional import cosine_similarity
        input_embedding = get_embedding(tmp_path, model, device)
        input_tensor = torch.tensor(input_embedding).unsqueeze(0)
        similarities = []
        for speaker_name, db_embedding in voice_db.items():
            db_tensor = torch.tensor(db_embedding).unsqueeze(0)
            similarity = cosine_similarity(input_tensor, db_tensor).item()
            similarities.append({
                "speaker": speaker_name,
                "similarity": round(float(similarity), 4),
                "confidence": round(min(100, max(0, similarity * 100)), 2)
            })
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        top_matches = similarities[:3]
        best_match = similarities[0]
        top2_similarity = similarities[1]["similarity"] if len(similarities) > 1 else -1.0
        margin = best_match["similarity"] - top2_similarity
        effective_threshold = threshold - 0.05 if margin >= 0.08 else threshold
        effective_threshold = max(0.55, effective_threshold)
        if best_match["similarity"] >= effective_threshold:
            return JSONResponse(content={
                "success": True,
                "identified": True,
                "speaker": best_match["speaker"],
                "similarity": best_match["similarity"],
                "threshold": threshold,
                "effective_threshold": round(float(effective_threshold), 4),
                "margin_vs_next": round(float(margin), 4),
                "top_matches": top_matches,
                "message": f"Identified as {best_match['speaker']}",
                "confidence": best_match["confidence"]
            })
        else:
            return JSONResponse(content={
                "success": True,
                "identified": False,
                "speaker": None,
                "similarity": best_match["similarity"],
                "threshold": threshold,
                "effective_threshold": round(float(effective_threshold), 4),
                "margin_vs_next": round(float(margin), 4),
                "top_matches": top_matches,
                "closest_speaker": best_match["speaker"],
                "message": "No matching speaker found in database. Please add this voice to the system.",
                "confidence": 0
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        os.unlink(tmp_path)

@app.post("/enroll")
async def enroll_speaker(audio: UploadFile = File(...), speaker_name: str = None):
    if not speaker_name or not speaker_name.strip():
        raise HTTPException(status_code=400, detail="Speaker name is required")
    if speaker_name in voice_db:
        raise HTTPException(status_code=400, detail=f"Speaker '{speaker_name}' already exists in database")
    if not audio.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.filename).suffix) as tmp_file:
        content = await audio.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    try:
        embedding = get_embedding(tmp_path, model, device)
        voice_db[speaker_name] = embedding
        with open("voice_db.pkl", "wb") as f:
            pickle.dump(voice_db, f)
        return JSONResponse(content={
            "success": True,
            "message": f"Speaker '{speaker_name}' enrolled successfully",
            "speaker": speaker_name,
            "total_speakers": len(voice_db)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment error: {str(e)}")
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
