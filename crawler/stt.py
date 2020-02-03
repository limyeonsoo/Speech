import sox
import pysox
import glob
import os

def transcribe_file(speech_file):
    """Transcribe the given audio file."""
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    client = speech.SpeechClient()

    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()

    audio = types.RecognitionAudio(content=content)
    config = types.RecognitionConfig(
 #       encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='ko-KR')
    response = client.recognize(config, audio)
    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    try:
        return(response.results[0].alternatives[0].transcript)
    except(IndexError):
        return("pass")

"""
    if response == '  ':
        print("pass")
    else:
        print(response.results[0].alternatives[0].transcript)
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print(u'Transcript: {}'.format(result.alternatives[0].transcript))
"""

keyword_dic = {}

"""
keyword_lists = glob.glob("/home/knlab/Summer_study/Lab_members/YS/honk/keyword_spotting_data_generator/data/*")
for keyword in keyword_lists:
    wav_lists = glob.glob(keyword+"/*")
    with open(keyword[82:]+".txt", "w") as f:
        if (len(wav_lists) > 0):
            keyword_dic[keyword[82:]]=len(wav_lists)
        for i, wave in enumerate( wav_lists):
            audio = pysox.CSoxStream(wave)
            print(audio)
            print(audio.get_signal())
            print("\n")
            f.write(str(i)+" : ")
            f.write(transcribe_file(wave))
            f.write("\n")
"""
print(type(transcribe_file("/mnt/sdb/YS/success/health/비타민/c_LQXZJ-14OM0_51..wav")))

