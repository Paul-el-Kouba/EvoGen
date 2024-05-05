import os
import pandas
import pickle
import subprocess
import numpy as np
from knn import KNN
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

CURRENT_DIR = os.getcwd()


class Sentiment_Learner:
    def __init__(self):
        pass

    def extract_features(self, song: str = None) -> None:
        """
        This function extract multiple features from either a set of midi files ({data_path}/midi),
        or a single midi file and saves them ({data_path}/midi/{song}) in both an .xml and a .csv files
        named as follows day-mon-year-hr-min-sec_feature_values

        :param song: str - path of a single midi file
        :param data_path: str - The path of the directory of midi files
        :return: None - saves a .csv , a .arff and a .xml file in
        """
        date_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S_")

        if song is None:
            # For entire folder

            subprocess.run(["java", "-Xmx6g", "-jar", "./jSymbolic_lib/jSymbolic_2_2_user/jSymbolic2.jar",
                            f"./midi_files/midi/", f"./midi_files/features/{date_time}feature_values.xml",
                            "./jSymbolic_lib/test_def.xml"])
        else:
            # For single File

            subprocess.run(["java", "-Xmx6g", "-jar", "./jSymbolic_lib/jSymbolic_2_2_user/jSymbolic2.jar",
                            f"{song}", f"./midi_files/features/{song[18:-4]}.xml",
                            "./jSymbolic_lib/test_def.xml"])

    def generate_new_set(self, features_path: str, labels_paths: str, learner_name='learner.tr') -> None:
        """
        This creates a new learner.tr.
        This file represents the trained model.

        :param learner_name: name of the tr file
        :param features_path: str - path of the feature_values.csv
        :param labels_paths: str - path of the labels.csv
        !! all the elements of features_values MUST be in labels.csv
        :return: None - saves a learner.tr on the local directory
        """
        features = {}
        labels = {}

        df = pandas.read_csv(features_path)
        df2 = pandas.read_csv(labels_paths)

        arr = list(df.to_numpy())
        brr = list(df2.to_numpy())

        for a in arr:
            features.update({a[0]: a[1:]})

        for b in brr:
            labels.update({b[0]: b[1:]})

        features = dict(sorted(features.items()))
        labels = dict(sorted(labels.items()))

        with open(f'{learner_name}', 'wb') as f:
            pickle.dump([features, labels], f)

    def load_learner(self, learner="learner.tr") -> List[dict]:
        """
        Load the learner.tr model

        :return: list - two dicts: [features, sentiments]
        """
        with open(f'{learner}', 'rb') as f:
            feats_labels = pickle.load(f)
        return feats_labels

    def predict_sentiments(self, midi_song: str) -> Dict[str, List[float]]:
        """

        :param midi_song: Format ./midi_files/midi/song_name.mid
        :return: List (1x5) containing the predicted emotions.
        """
        # Import trained values
        trainer = self.load_learner("learner_full.tr")

        # Extract features and sentiments
        labelled_features = list(trainer[0].values())
        labelled_features = list(np.asarray(labelled_features).astype(float))
        labelled_sentiments = trainer[1]

        self.extract_features(midi_song)

        generated_features = pd.read_csv(f'./midi_files/features/{midi_song[18:-4]}.csv')
        #generated_features = generated_features.fillna(0)
        generated_features = generated_features.to_numpy()[0][1:]
        arr = generated_features
        generated_features = np.where(arr != ' NaN', arr.astype(float), 0)

        model = KNN(21)
        model.fit(labelled_features, list(labelled_sentiments.keys()))
        NN_pieces = model.predict(generated_features)

        NN_sentiment = [labelled_sentiments[i] for i in NN_pieces]

        return {f"{midi_song[18:-4]}.mid": list(np.average(NN_sentiment, axis=0))}
