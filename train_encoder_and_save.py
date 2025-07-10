# train_encoder_and_save.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your data (adjust the path if necessary)
visit_df = pd.read_csv('data/air_visit_data.csv')

# Create and fit the LabelEncoder
le = LabelEncoder()
visit_df['store_id_encoded'] = le.fit_transform(visit_df['air_store_id'])

# Save the encoder to a file
joblib.dump(le, 'models/store_label_encoder.pkl')

print("Label encoder saved as 'models/store_label_encoder.pkl'")
