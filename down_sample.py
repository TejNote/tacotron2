import librosa
import soundfile as sf
import os

def down_sample(input_wav, resample_sr):

    if not os.path.isfile(input_wav):
        print(":::::::::::::: Error " + input_wav + "::::::::::::::")
        return
    _, filename = os.path.split(input_wav)
    y, sr = librosa.load(input_wav)
    sf.write('resampled/' + filename, y, resample_sr, format='WAV', endian='LITTLE', subtype='PCM_16')


target_sr = 22050
# down_sample("/home/docfriends/문서/tacotron2_nvidia/kss/wavs/4_5191.wav", target_sr)
f = open("filelists/kss_train_filelist_v3_temp.txt", 'r')

isStart = False
while True:
    line = f.readline()
    if not line: break
    result = line.split("|")
    if result[0].find("4_5083.wav") > -1:
        isStart = True

    if isStart:
        path = "/home/docfriends/문서/tacotron2_nvidia/" + result[0].replace("\n", "").replace("./", "")
        down_sample(path, target_sr)

f.close()

#/home/docfriends/문서/tacotron2_nvidia/kss/wavs/1/1_0781.wav
#
