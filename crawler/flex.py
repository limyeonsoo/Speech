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
        return("Out")


"""
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print(u'Transcript: {}'.format(result.alternatives[0].transcript))
    return response.result.alternatives[0].transcript
"""
import os
from tqdm import tqdm
import subprocess

cmd = "ffmpeg -i "

for root, dirs, files in os.walk("/home/knlab/Summer_study/Lab_members/YS/honk/keyword_spotting_data_generator/data"):
    for file in tqdm(files):
        if os.path.isfile(os.path.join(root,file).replace("wav","flac")):
            continue
        label = root.split('/')[-1]
        subprocess.call(cmd+os.path.join(root, file)+' '+os.path.join(root, file).replace("wav","flac"),shell=True)


import os
from tqdm import tqdm
import subprocess
import glob

cmd = "ffmpeg -i "
keyword_len = {}
keyword_dic = {}

for root, dirs, files in os.walk("/home/knlab/Summer_study/Lab_members/YS/honk/keyword_spotting_data_generator/data"):
    label = root.split('/')[-1]
    temp = []
    keyword_len[label]=len(files)
    for file in tqdm(files):
        if file[-5:] == ".flac":
            print(file)
            temp.append(str(transcribe_file(os.path.join(root,file))))
            print(str(transcribe_file(os.path.join(root,file))))
            keyword_dic[label] = temp
with open("result.txt","w") as f:
    for word in keyword_len.items():
        f.write(str(word[0])+": "+str(word[1]))
        f.write("\n")
    f.write("\n")
    for word in keyword_dic.items():
        f.write(str(word[0])+": "+str(word[1]))
        f.write("\n")
