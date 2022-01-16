import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import json

dataset_path = "D:/MCSC/ML/Project/test_data/"
json_path = "data.json"
duration = 10 #measured in seconds
sample_rate = 22050
samples_per_track = sample_rate * duration
def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length = 512, num_segments=5):

    data = {"mappings" : [],    #name of the parts
            "mfcc" : [],
            "labels" : []
    }

    num_samples_per_segment = int(samples_per_track / num_segments)

    mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:
            dirpath_component = dirpath.split('/')
            label = dirpath_component[-1]

            data["mappings"].append(label)
            print("\nProcessing {}".format(label))

            for f in filenames: #process audio files for specific parts
                file_path = os.path.join(dirpath, f) #load audio file of machine parts

                signal, sr = librosa.load(file_path, sr = sample_rate)

                for s in range(num_segments): #process segments extracting mfcc
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
                    mfcc = mfcc.T

                    #store mfcc for segment if it has expected length
                    if len(mfcc) == mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(dataset_path, json_path)




