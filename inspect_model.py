import joblib
import pandas as pd
from pathlib import Path
import json

base = Path('.')
model = joblib.load(base/'placement_model.pkl')
scaler = joblib.load(base/'scaler.pkl')
df = pd.read_csv(base/'student_placement_prediction_dataset_2026.csv')

print('model_type', type(model).__name__)
print('n_features_in_', getattr(model, 'n_features_in_', None))
print('classes_', getattr(model, 'classes_', None))
print('scaler_type', type(scaler).__name__)
print('df_columns', list(df.columns))
print(df.head().to_string())

nb = json.loads(Path('Student.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'LabelEncoder' in src or 'StandardScaler' in src or 'joblib.dump' in src or 'train_test_split' in src or 'X_train' in src:
            print('\n=== CELL', i, '===')
            print(src)
