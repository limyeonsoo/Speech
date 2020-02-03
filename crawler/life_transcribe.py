import os
from tqdm import tqdm
import json
from collections import defaultdict
from time import sleep
import psutil

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
keyword_success = {}

f = open("/home/knlab/Summer_study/Lab_members/YS/life_result.json","w")
f.close()
f = open("/home/knlab/Summer_study/Lab_members/YS/life_result_count.json","w")
f.close()
count = 0
error = []
for root, dirs, files in os.walk("/mnt/sdb/YS/youtube_data/일상생활/"):
    f1 = open("/home/knlab/Summer_study/Lab_members/YS/life_result.json","a")
    f2 = open("/home/knlab/Summer_study/Lab_members/YS/life_result_count.json","a")
    count+=1
    success_count = 0
    keyword_result = defaultdict(str)
    keyword_success = {}
    temp = []
    label = root.split('/')[-1]
    for file in tqdm(files):
        try:
            result = str(transcribe_file(os.path.join(root+"/"+file)))
        except OSError:
            error.append((label,file))
            result = "error"
            print("OSError : ",label, file)
            pass
        except:
            error.append((label,file))
            result = "error"
            print("Error : ",label, file)
            pass
        temp.append(result)
        if label == result:
            success_count+=1
    print(count) 
    keyword_result[label]=(temp, success_count, len(files))
    keyword_success[label]=(success_count,len(files))
    f1.write(json.dumps(keyword_result,ensure_ascii = False))
    f1.write("\n")
    f2.write(json.dumps(keyword_success, ensure_ascii = False))
    f2.write("\n")
    f1.close()
    f2.close()
