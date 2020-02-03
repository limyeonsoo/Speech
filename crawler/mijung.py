import os
from tqdm import tqdm
import json
from time import sleep

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
        return(" * ")
    
keyword_result = {}
f = open("/home/knlab/Summer_study/Lab_members/YS/result3.txt","w")
for root, dirs, files in os.walk("/mnt/sdb/YS/youtube_data/?/"):
    for directory in tqdm(dirs[120:]):
        #temp = []
        count = 0
        #keyword_result[directory] = {}
        f.write(directory+" : ")
        for file in os.listdir(root+directory):
            temp = str(transcribe_file(os.path.join(root,directory+"/"+file)))
            sleep(0.5)
            if temp == directory: count+=1
            f.write(" "+temp+" ")
            #f.write(" "+str(transcribe_file(os.path.join(root,directory+"/"+file))))
        f.write("  ("+str(count)+"/"+str(len(os.listdir(root+directory)))+")  "+"\n\n")
            #keyword_result[directory] += str(transcribe_file(os.path.join(root,directory+"/"+file)))
            #temp.append(str(transcribe_file(os.path.join(root,directory+"/"+file))))
        #keyword_result[directory] = temp
        #f.write(directory+" : "+temp)
          

            

            
