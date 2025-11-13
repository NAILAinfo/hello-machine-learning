import numpy as np
import tensorflow as tf
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Set random seed
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted",
    "num_root","num_file_creations","num_shells","num_access_files",
    "num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate",
    "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label"
]
df = pd.read_csv(url, names=columns)
print(df.head())# Afficher les 5 premières lignes
print("Shape of dataset:", df.shape)
print(df['label'].value_counts())
df = df.drop(columns=['diff_srv_rate', 'srv_diff_host_rate', 'dst_host_srv_diff_host_rate','dst_host_diff_srv_rate'])
# X : toutes les colonnes sauf 'label'
X = df.drop(columns=['label'])
# y : la colonne 'label' uniquement
y = df['label']
# Sélectionner les colonnes catégorielles à encoder
categorical_cols = ['protocol_type', 'service', 'flag']
# Encoder ces colonnes en variables dummy
X_encoded = pd.get_dummies(X, columns=categorical_cols)
y_binary = y.apply(lambda x: 0 if x == 'normal' else 1)
# Identifier les colonnes numériques (toutes sauf les colonnes encodées en dummy)
numeric_cols = X_encoded.select_dtypes(include=['int64', 'float64']).columns
# Créer le scaler et l'appliquer
scaler = StandardScaler()
X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,        # features
    y_binary,         # labels binaires
    test_size=0.2,    # 20% pour le test
    random_state=42,  # pour la reproductibilité
    stratify=y_binary # conserve la proportion normal/attaque
)