import whisperx
from machine_setup import get_optimal_device,get_optimal_compute_type

hf_access_token = ""

device = get_optimal_device()
audio_file = "media\\audio.wav"
batch_size = 16 #reduce if low gpu mem
compute_type = get_optimal_compute_type()

min_speakers : None | int = None
max_speakers : None | int = None



#load target audio
audio = whisperx.load_audio(audio_file)

#transcribe
model_transcribe = whisperx.load_model("base",device,compute_type=compute_type)
result = model_transcribe.transcribe(audio=audio,batch_size=batch_size)


#align whisper output
model_alignment, alignment_metadata = whisperx.load_align_model(language_code=result["language"],device=device)
result = whisperx.align(result["segments"], model=model_alignment, align_model_metadata=alignment_metadata, audio=audio, device=device, return_char_alignments=False)


#diarize
model_diarize = whisperx.DiarizationPipeline(use_auth_token=hf_access_token, device=device)
diarized_segments = model_diarize(audio=audio,min_speakers=min_speakers,max_speakers=max_speakers)

whisperx.assign_word_speakers(diarized_segments,result)

print(result["segments"])