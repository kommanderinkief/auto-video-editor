from audio import transcribe_diarized
from machine import clear_gpu
import json
import torchaudio
from typing import Union

torchaudio.set_audio_backend("soundfile")

hf_access_token = ""  # https://huggingface.co/settings/tokens

audio_file = "AudioRecording.wav"

batch_size = 16  # reduce if low GPU mem / increase if high
compute_type = "float32"
whisper_model = "large-v2"
language = "en"

min_speakers: Union[None, int] = 2
max_speakers: Union[None, int] = 2

clear_gpu()
diarized_transcription = transcribe_diarized(
    audio_filepath=audio_file,
    hf_access_token=hf_access_token,
    whisper_model=whisper_model,
    batch_size=batch_size,
    min_speakers=min_speakers,
    max_speakers=max_speakers,
    compute_type=compute_type,
    debug_mode=True,
)

print(diarized_transcription)
with open("sample.json", "w") as f:
    f.write(json.dumps(diarized_transcription, indent=2))
