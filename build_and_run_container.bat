@echo off
docker stop whisper-loop-app && docker rm whisper-loop-app
docker rmi whisper-loop-app
docker build -t whisper-loop-app . && ^
docker run -d -it ^
-v %cd%:/app ^
--name whisper-loop-app ^
whisper-loop-app