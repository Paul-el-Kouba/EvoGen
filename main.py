import os
import shutil
import subprocess
from similarity_measures import TS_SS
from sentiment_learner import Sentiment_Learner


def empty_directory():
    directory = "./midi_files/midi/"
    directory2 = "./midi_files/features/"
    shutil.rmtree(directory)
    shutil.rmtree(directory2)

    os.makedirs(directory)
    os.makedirs(directory2)


empty_directory()

target_emotions = []
print("Welcome to EvoGen-Music composer.")
target_emotions.append(float(input("Please input anger: ")))
target_emotions.append(float(input("Please input joy: ")))
target_emotions.append(float(input("Please input love: ")))
target_emotions.append(float(input("Please input sadness: ")))
target_emotions.append(float(input("Please input surprise: ")))


# predicted_emotions = sl.predict_sentiments("./midi_files/midi/plzwork.mid")
# print(predicted_emotions)

"""
    Generate the pool of themes
"""
try:
    command = ["python3.7", "generate.py", "models/LMD2/", "../midi_files/midi/", "--n", "100", "--no_audio"]
    cwd = "./polyphemus/"
    subprocess.run(command, cwd=cwd)
except subprocess.CalledProcessError as e:
    # Handle any errors that occur during command execution
    print("Error:", e)

"""
    Evaluate the pool of themes and retain the top 50
"""
sentimentLearner = Sentiment_Learner()
similarityMeasure = TS_SS()
midi_dir = "./midi_files/midi/"

file_emotion_dict = {}

for file in os.listdir(midi_dir):
    if file.endswith(".mid"):
        midi_file = os.path.join(midi_dir, file)
        theme, predicted = sentimentLearner.predict_sentiments(midi_file)
        file_emotion_dict[theme] = predicted

distances = {file: similarityMeasure.euclidean(target_emotions, predicted_emotions) for file, predicted_emotions in file_emotion_dict.items()}
sorted_files = sorted(distances.items(), key=lambda x: x[1])
closest_files = dict(sorted_files[:50])
for file in file_emotion_dict.keys():
    if file not in closest_files:
        os.remove(file)
        del file_emotion_dict[file]


"""
    Expand on the themes
"""



"""
    Choose the top 'n' pieces and give them to the user
"""