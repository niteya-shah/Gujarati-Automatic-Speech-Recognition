# Automatic Speech Recognition for Gujarati Language

Code to Train a Model for gujarati text to speech using the Microsoft Research Open Data

https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e

This Model can be used to recognize any language as long the necessary changes are made to Corpora.py file.

### MOST OF THE DIRECTORY LINKS ARE ABSOLUTE TO MY COMPUTER AND NEED TO BE CHANGED

## Setup
Ensure that the Following File Structure is present

-Root

---gu-in-Test

-------/transcription.csv

-------/Audios

---gu-in-Train

-------/Audios

-------/transcription.csv

## Initialize
```

python read_dataset.py

```
## Training 
```
python model_train.py
```
## Testing
```

python test_model.py

```

## License
[MIT](https://choosealicense.com/licenses/mit/)
