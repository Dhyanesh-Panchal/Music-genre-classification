import pandas as pd
import librosa
import os
import IPython.display as ipd
from IPython.display import Audio
import numpy as np
from tqdm import tqdm as progressbar
import json
import multiprocessing as mp
print(os.getcwd())


def generate_filepath(track_id):
    formatted_track_id = str(track_id).zfill(6)
    path = f"./Music-genre-classification/fma_small/fma_small/{formatted_track_id[0:3]}/{formatted_track_id}.mp3"
    return path

def rms_energy(audio):
    energy = np.sum(np.square(audio)) / len(audio)   
    # Calculate RMS
    rms = np.sqrt(energy)
    return rms
def compute_mfcc(audio, sr, n_mfcc=10):
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_stats = []
    for coeff in mfccs:
        mfcc_stats.append((np.mean(coeff),np.var(coeff)))
    return mfcc_stats
def spectral_centroid(audio, sr):
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    return np.mean(spectral_centroid),np.var(spectral_centroid)

def spectral_bandwidth(audio, sr):
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    
    return np.mean(spectral_bandwidth),np.var(spectral_bandwidth)

def spectral_contrast(audio, sr):
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    
    return np.mean(spectral_contrast),np.var(spectral_contrast)

def spectral_rolloff(audio, sr):
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    
    return np.mean(spectral_rolloff),np.var(spectral_rolloff)
def zero_crossing_rate(audio):
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
    
    return np.mean(zero_crossing_rate),np.var(zero_crossing_rate)

def chroma_features(audio, sr):
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    
    return np.mean(chroma),np.var(chroma)

def tonnetz(audio, sr):
    tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
    
    return np.mean(tonnetz),np.var(tonnetz)



def prepare_features(track,id):
    sr=track["bit_rate"]
    path=track["file_path"]
    audio,sr = librosa.load(path,sr=sr)
    features = {}
    features["track_id"]=track["track_id"]
    features["rms_energy"]=rms_energy(audio)
    mfcc_features = compute_mfcc(audio,sr)
    for index,mfcc_vector in enumerate(mfcc_features):
        features[f"mfcc_{index+1}_mean"],features[f"mfcc_{index+1}_var"] = mfcc_vector
    features["spectral_centroid_mean"],features["spectral_centroid_var"] = spectral_centroid(audio,sr)
    features["spectral_bandwidth_mean"],features["spectral_bandwidth_var"] = spectral_bandwidth(audio,sr)
    features["spectral_contrast_mean"],features["spectral_contrast_var"] = spectral_contrast(audio,sr)
    features["spectral_rolloff_mean"],features["spectral_rolloff_var"] = spectral_rolloff(audio,sr)
    features["zero_crossing_rate_mean"],features["zero_crossing_rate_var"] = zero_crossing_rate(audio)
    features["chroma_features_mean"],features["chroma_features_var"] = chroma_features(audio,sr)
    features["tonnetz_mean"],features["tonnetz_var"] = tonnetz(audio,sr)
    try:
        data = pd.read_csv(f"./Music-genre-classification/Features/preprocess_batch_{id}.csv")
        data = pd.concat([data,pd.DataFrame([features])],axis=0)
        data.to_csv(f"./Music-genre-classification/Features/preprocess_batch_{id}.csv",index=False)
    except Exception as e:

        print(e)
        data = pd.DataFrame([features])
        data.to_csv(f"./Music-genre-classification/Features/preprocess_batch_{id}.csv",index=False)


def prepare_data_wrapper(df,id):
    print(f"For Process #{id}: {len(df)}")
    for index, track in progressbar(df.iterrows(),total=len(df),desc=f'Process #{id}'):
        prepare_features(track,id)

if __name__ == "__main__":
    data_mapping = pd.read_csv("./Music-genre-classification/data_labels_relation.csv")

    data_mapping["file_path"] = data_mapping["track_id"].apply(generate_filepath)

    process_1 = mp.Process(target=prepare_data_wrapper,args=(data_mapping[:2000],1))
    process_2 = mp.Process(target=prepare_data_wrapper,args=(data_mapping[2000:4000],2))
    process_3 = mp.Process(target=prepare_data_wrapper,args=(data_mapping[4000:6000],3))
    process_4 = mp.Process(target=prepare_data_wrapper,args=(data_mapping[6000:8000],4))
            

    process_1.start()
    process_2.start()
    process_3.start()
    process_4.start()

    process_1.join()
    process_2.join()
    process_3.join()
    process_4.join()