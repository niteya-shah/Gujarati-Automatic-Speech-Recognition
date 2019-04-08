Automatic Speech Recognition for Gujarati Language

This Model can be used to recognize any language as long the necessory changes are made to Corpora.py file.

The Current File Structure expects there to be 

-Root
---gu-in-Test

-------/transcription.csv
-------/Audios
---gu-in-Train
-------/Audios
-------/transcription.csv

Where Audios folders need to have the audio files , in a format that can be read by librosa.load
The Transcription need to follow the following format

000010001	?????? ????????? ???? ?????? ?? ??? ??
000010002	? ??????? ??????????? ????? ??????? ????? ??? ????????? ?????????? ??????????? ?? ????? ???
000010003	?????? ????? ?????? ????? ??????? ???? ???????? ??? ? ??????? ??
000010004	???????? ????? ?????? ????? ??????? ???? ???????? ??? ? ??????? ??
000010005	???????? ????? ????? ????????? ???????? ?? ??????????? ????? ????????? ?????? ????
