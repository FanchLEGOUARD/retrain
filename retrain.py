import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import re
import numpy as np
from tqdm.notebook import tqdm
from itertools import product
import time
import joblib
from dotenv import load_dotenv
import os

load_dotenv()

HOST_MONGODB = os.getenv("HOST_MONGODB")
MONGO_DB_APPNAME = os.getenv("MONGO_DB_APPNAME")
PASSWORD_MONGODB = os.getenv("PASSWORD_MONGODB")
USER_MONGODB = os.getenv("USER_MONGODB")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DB_NAME = os.getenv("DB_NAME")


