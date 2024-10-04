import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydub import AudioSegment
from scipy.signal import lfilter
import uvicorn

app = FastAPI()

def add_reverb(audio, reverb_delay=0.1, decay=0.5):
    # Convert audio to numpy array
    samples = np.array(audio.get_array_of_samples()).astype(np.float64)
    
    # Create reverb effect
    reverb_samples = int(reverb_delay * audio.frame_rate)
    reverb = np.zeros_like(samples)
    reverb[reverb_samples:] = samples[:-reverb_samples]
    reverb = reverb * decay
    
    # Mix original audio with reverb
    processed = samples + reverb
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(processed))
    if max_val > 0:
        processed = processed / max_val * 32767
    
    # Convert back to int16
    processed = processed.astype(np.int16)
    
    return AudioSegment(
        processed.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=2,
        channels=audio.channels
    )

def slow_down(audio, slow_factor=0.8):
    # Slow down the audio by resampling
    slowed = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * slow_factor)
    })
    return slowed.set_frame_rate(audio.frame_rate)

@app.post("/process_audio/")
async def process_audio(
    file: UploadFile = File(...),
    reverb_delay: float = Form(0.1),
    reverb_decay: float = Form(0.5),
    slow_factor: float = Form(0.8)
    
):
    # Read the uploaded file
    content = await file.read()
    audio = AudioSegment.from_file(io.BytesIO(content), format=file.filename.split('.')[-1])

    # Apply effects
    reverb_audio = add_reverb(audio, reverb_delay=reverb_delay, decay=reverb_decay)
    slowed_reverb_audio = slow_down(reverb_audio, slow_factor=slow_factor)

    # Export the processed audio
    buffer = io.BytesIO()
    slowed_reverb_audio.export(buffer, format="mp3")
    buffer.seek(0)

    # Return the processed audio as a streaming response
    return StreamingResponse(buffer, media_type="audio/mpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)