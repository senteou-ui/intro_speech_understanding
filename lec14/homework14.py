import gtts
import speech_recognition
import librosa
import soundfile

def synthesize(text, lang, filename):
    tts = gtts.gTTS(text=text, lang=lang)
    tts.save(filename)

def make_a_corpus(texts, languages, filenames):
    recognized_texts = []
    r = speech_recognition.Recognizer()
    
    for text, lang, filename in zip(texts, languages, filenames):
        mp3_file = filename + ".mp3"
        wav_file = filename + ".wav"
        
        
        synthesize(text, lang, mp3_file)
        
        
        data, samplerate = librosa.load(mp3_file)
        soundfile.write(wav_file, data, samplerate)
        
      
        with speech_recognition.AudioFile(wav_file) as source:
            audio = r.record(source)
            recognized_text = r.recognize_google(audio, language=lang)
            recognized_texts.append(recognized_text)
            
    return recognized_texts