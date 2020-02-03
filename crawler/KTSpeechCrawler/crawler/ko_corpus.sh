#!/bin/bash
trap "exit" INT

target_dir=$1
filter_dir=$2

declare -a arr=("센트럴 인버터","스트링 인버터","인버터","태양광 모듈","태양광","모듈","집적판","능동 디바이스","발전 효율","발전 전력","주파수 변동", "시각화", "모니터링", "원격관제", "모듈상태", "발전량 비교", "구역별 발전량 비교", "열화상 카메라", "장애시간", "드론", "이상온도","에너지저장장치", "전력저장장치", "고장진단","출력예측","배터리 화재", "각도 제어", "각도제어", "출력제어", "출력 제어","발전" )
for i in `seq 1 30`;
do
    echo "PAGE="$i
    for kw in "${arr[@]}"
    do
        youtube-dl --download-archive ./ko-downloaded.txt --no-overwrites -f mp4 --restrict-filenames --youtube-skip-dash-manifest --prefer-ffmpeg --socket-timeout 20  -iwc --write-info-json -k --write-srt --sub-format ttml --sub-lang ko --convert-subs vtt  "https://www.youtube.com/results?sp=EgQIBCgB&q="$kw"&p="$i -o "$target_dir%(id)s%(title)s.%(ext)s" --exec "python3 ./crawler/process.py {} '$filter_dir'"
    done
done
