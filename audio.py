"""interaction with audio"""

import whisperx
from pydub import AudioSegment
import pathlib
from whisperx import types as whisperx_types
from machine import get_optimal_device,get_optimal_compute_type,clear_gpu,T_Device,T_Compute_Type,T_Model

def transcribe(
        audio_filepath:str,
        whisper_model: T_Model,
        language:str | None = None,
        batch_size: int = 16,
        device: T_Device | None = None,
        compute_type: T_Compute_Type | None = None,
        debug_mode: bool = False,
    ) -> whisperx_types.AlignedTranscriptionResult :
        """transcribe the target audio"""

        ## assign default arguments

        if device == None:
            device = get_optimal_device()

        if compute_type == None:
            compute_type = get_optimal_compute_type()


        #load target audio
        audio = whisperx.load_audio(audio_filepath)

        #transcribe
        model_transcribe = whisperx.load_model(whisper_model,device,compute_type=compute_type,language=language)
        result = model_transcribe.transcribe(audio=audio,batch_size=batch_size)

        if debug_mode:
            print("TRANSCRIBED")
        clear_gpu()


        #align whisper output
        model_alignment, alignment_metadata = whisperx.load_align_model(language_code=result["language"],device=device)
        result = whisperx.align(result["segments"], model=model_alignment, align_model_metadata=alignment_metadata, audio=audio, device=device, return_char_alignments=False)

        if debug_mode:
            print("ALIGNED")
        clear_gpu()

        return result


def diarize(
        audio_filepath:str,
        transcription_result:whisperx_types.AlignedTranscriptionResult | whisperx_types.TranscriptionResult,
        hf_access_token:str,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        device: T_Device | None = None,
        debug_mode: bool = False,
    ):
        """diarize an already transcribed audio-file"""
        
        if device == None:
            device = get_optimal_device()

        #load target audio
        audio = whisperx.load_audio(audio_filepath)

        #diarize
        model_diarize = whisperx.DiarizationPipeline(use_auth_token=hf_access_token, device=device)
        diarized_segments = model_diarize(audio=audio,min_speakers=min_speakers,max_speakers=max_speakers)

        if debug_mode:
            print("DIARIZED")

        clear_gpu()

        whisperx.assign_word_speakers(diarized_segments,transcription_result)

        return transcription_result


def transcribe_diarized(
        audio_filepath:str,
        hf_access_token:str,
        whisper_model: T_Model = "medium",
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        batch_size: int = 16,
        device: T_Device | None = None,
        compute_type: T_Compute_Type | None = None,
        debug_mode: bool = False,
    ) -> whisperx_types.AlignedTranscriptionResult:
        """transcribe & diarize the specified audio-file"""
        
        ## default args
        if device == None:
            device = get_optimal_device()

        if compute_type == None:
            compute_type = get_optimal_compute_type()

        #transcribe
        aligned_transcription = transcribe(
             audio_filepath=audio_filepath,
             whisper_model=whisper_model,
             batch_size=batch_size,
             device=device,
             compute_type=compute_type,
             debug_mode=debug_mode,
        )

        
        #diarize
        diarized_transcription = diarize(
             audio_filepath=audio_filepath,
             transcription_result=aligned_transcription,
             hf_access_token=hf_access_token,
             min_speakers=min_speakers,
             max_speakers=max_speakers,
             device=device,
             debug_mode=debug_mode,
        )


        return diarized_transcription

# TODO
# create accurate return typings which accounts for implementation of 'speaker' attribute, post diarization.

def filter_segments_to_only_speaker(speaker_id:str,diarized_transcription:whisperx_types.AlignedTranscriptionResult | whisperx_types.TranscriptionResult):
    """select only segments in which the speaker specified by speaker_id speaks within the passed diarized transcription"""

    filter_lambda = lambda segment: "speaker" in segment.keys() and segment["speaker"] == speaker_id
    
    filtered =  filter(filter_lambda,diarized_transcription["segments"])
    
    return list(filtered)


def extract_speaker_segments_to(speaker_id:str,diarized_transcription:whisperx_types.AlignedTranscriptionResult | whisperx_types.TranscriptionResult,originating_audio_clip_filepath:str,out_dir:str) -> list[str]:
    """save all audio clips in which the specified speaker is speaking, to a specified directory. Returns a list of all filepaths of created audio clips"""
    
    #fetch target segments (the segments in which the specified speaker is speaking)
    target_segments = filter_segments_to_only_speaker(speaker_id=speaker_id,diarized_transcription=diarized_transcription)
    
    #load audio
    audio = AudioSegment.from_file(originating_audio_clip_filepath)
    
    #transform out_dir to absolute path
    out_dir = str(pathlib.Path(out_dir).resolve())

    #create the specified directory if does not already exist
    pathlib.Path(out_dir).mkdir(parents=True,exist_ok=True)

    #generate sub-clips and export to specified directory
    generated_filepaths = []
    for i,segment in enumerate(target_segments):
        start = segment["start"] * 1000
        end = segment["end"] * 1000

        trimmed_audio : AudioSegment = audio[start:end]

        format = "mp3"
        filename = f"clip-{segment['speaker']}-{i:03}" + f".{format}"
        filepath = str(pathlib.Path(out_dir).joinpath(filename))

        trimmed_audio.export(filepath,format=format)

        generated_filepaths.append(filepath)

    return generated_filepaths
        
         
     





     
     