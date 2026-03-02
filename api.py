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

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class SpeakerCNN(nn.Module):
    def __init__(self, num_classes, embedding_dim=256):
        super(SpeakerCNN, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_layer = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        embedding = torch.relu(self.embedding_layer(x))
        output = self.classifier(embedding)
        return output, embedding


# ============================================================
# AUDIO PROCESSING FUNCTIONS
# ============================================================
def audio_to_melspectrogram(file_path, sample_rate=16000, n_mels=128, fixed_length=128):
    """Convert audio to mel spectrogram"""
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=512,
        hop_length=256
    )
    
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    if log_mel_spec.shape[1] < fixed_length:
        pad_width = fixed_length - log_mel_spec.shape[1]
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel_spec = log_mel_spec[:, :fixed_length]
    
    return log_mel_spec.astype(np.float32)


def get_embedding(file_path, model, device):
    """Extract 256-d embedding from audio file"""
    spec = audio_to_melspectrogram(file_path)
    spec_tensor = torch.tensor(spec).unsqueeze(0).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        _, embedding = model(spec_tensor)
    
    return embedding.squeeze(0).cpu().numpy()


def identify_speaker(audio_path, model, voice_db, device, threshold=0.80):
    """Identify speaker using cosine similarity"""
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


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Speaker Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
voice_db = None
NUM_CLASSES = None


@app.on_event("startup")
async def load_model():
    """Load model and voice database on startup"""
    global model, voice_db, NUM_CLASSES
    
    try:
        with open("voice_db.pkl", "rb") as f:
            voice_db = pickle.load(f)
        
        NUM_CLASSES = len(voice_db)
        print(f"✓ Voice database loaded: {NUM_CLASSES} speakers")
        print(f"  Speakers: {list(voice_db.keys())}")
        
        model = SpeakerCNN(num_classes=NUM_CLASSES, embedding_dim=256).to(device)
        model.load_state_dict(torch.load("model.pth", map_location=device))
        model.eval()
        print(f"✓ Model loaded successfully on {device}")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "device": str(device),
        "speakers": list(voice_db.keys()) if voice_db else [],
        "num_speakers": NUM_CLASSES
    }


@app.get("/speakers")
async def get_speakers():
    """Get list of enrolled speakers"""
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
    threshold: float = 0.75
):
    """Verify if uploaded audio matches the claimed speaker"""
    
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
        
        # Extract embedding from uploaded audio
        input_embedding = get_embedding(tmp_path, model, device)
        input_tensor = torch.tensor(input_embedding).unsqueeze(0)
        
        # Compare with claimed speaker's embedding
        claimed_db_embedding = voice_db[speaker]
        claimed_tensor = torch.tensor(claimed_db_embedding).unsqueeze(0)
        similarity = cosine_similarity(input_tensor, claimed_tensor).item()
        
        authenticated = similarity >= threshold
        
        return JSONResponse(content={
            "success": True,
            "authenticated": authenticated,
            "similarity": round(float(similarity), 4),
            "claimed_speaker": speaker,
            "threshold": threshold,
            "decision": "AUTHENTICATED ✅" if authenticated else "ACCESS DENIED ❌",
            "confidence": round(max(0, min(100, (similarity - threshold + 0.3) / 0.3 * 100)), 2)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        os.unlink(tmp_path)


@app.post("/verify")
async def verify_speaker(audio: UploadFile = File(...), threshold: float = 0.65):
    """Identify speaker from uploaded audio file (speaker identification mode)"""
    
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
        
        # Extract embedding from uploaded audio
        input_embedding = get_embedding(tmp_path, model, device)
        input_tensor = torch.tensor(input_embedding).unsqueeze(0)
        
        # Find best matching speaker
        best_speaker = None
        best_similarity = -1
        
        for speaker_name, db_embedding in voice_db.items():
            db_tensor = torch.tensor(db_embedding).unsqueeze(0)
            similarity = cosine_similarity(input_tensor, db_tensor).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker_name
        
        # Check if match exceeds threshold
        if best_similarity >= threshold:
            return JSONResponse(content={
                "success": True,
                "identified": True,
                "speaker": best_speaker,
                "similarity": round(float(best_similarity), 4),
                "threshold": threshold,
                "message": f"Identified as {best_speaker}",
                "confidence": round(min(100, (best_similarity - threshold + 0.3) / 0.3 * 100), 2)
            })
        else:
            return JSONResponse(content={
                "success": True,
                "identified": False,
                "speaker": None,
                "similarity": round(float(best_similarity), 4),
                "threshold": threshold,
                "closest_speaker": best_speaker,
                "message": "No matching speaker found in database. Please add this voice to the system.",
                "confidence": 0
            })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        os.unlink(tmp_path)


@app.post("/enroll")
async def enroll_speaker(audio: UploadFile = File(...), speaker_name: str = None):
    """Enroll a new speaker to the database"""
    
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
        # Extract embedding from uploaded audio
        embedding = get_embedding(tmp_path, model, device)
        
        # Add to voice database
        voice_db[speaker_name] = embedding
        
        # Save updated database to file
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
    uvicorn.run(app, host="0.0.0.0", port=8000)