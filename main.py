from audio import transcribe_diarized

hf_access_token = "" #https://huggingface.co/settings/tokens

audio_file = "media\\audio.wav"
batch_size = 16 #reduce if low gpu mem / increase if high
compute_type = "float32"

min_speakers : None | int = None
max_speakers : None | int = None

diarized_transcription = transcribe_diarized(audio_filepath=audio_file,hf_access_token=hf_access_token,batch_size=batch_size,min_speakers=min_speakers,max_speakers=max_speakers,compute_type=compute_type,debug_mode=True)

print(diarized_transcription)