from scipy.io import loadmat
import pandas as pd
import numpy as np
from sklearn.naive_bayes import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, average_precision_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

lr_model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced",max_iter=1500, random_state=
