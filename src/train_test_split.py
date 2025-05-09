import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import os
from zipfile import ZipFile
from sklearn.preprocessing import RobustScaler

script_path = os.path.abspath(__file__)

project_root = script_path
while not os.path.isdir(os.path.join(project_root, 'data')):
    new_root = os.path.dirname(project_root)
    if new_root == project_root: 
        raise Exception('El directorio raiz no se ha encontrado.')
    project_root = new_root

in_filepath = os.path.join(project_root, 'data', 'in', 'creditcard.csv')
out_filepath = os.path.join(project_root, 'data','out')
os.makedirs(out_filepath, exist_ok=True)

# Cargar el conjunto de datos
df_raw = pd.read_csv(in_filepath)

# Escalar los datos 
#features = df_raw.drop(columns='Class')
target = 'Class'

# scaler = RobustScaler()
# features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# Volver a unir con la variable objetivo
# df_scaled = pd.concat([features_scaled, target], axis=1)

# Primera división: 85% train_val, 15% test
split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
for train_val_ids, test_ids in split1.split(df_raw, df_raw['Class']):
    train_val_df = df_raw.loc[train_val_ids].reset_index(drop=True)
    test_df = df_raw.loc[test_ids].reset_index(drop=True)

# Segunda división: de train_val, 15% val (lo que da 70% train, 15% val del total)
split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)  # 0.1765 ≈ 15/85
for train_ids, val_ids in split2.split(train_val_df, train_val_df['Class']):
    train_df = train_val_df.loc[train_ids].reset_index(drop=True)
    val_df = train_val_df.loc[val_ids].reset_index(drop=True)

scaler = RobustScaler()
features_train = train_df.drop(columns='Class')
features_scaled_t = pd.DataFrame(scaler.fit_transform(features_train), columns=features_train.columns)
train_df_scaled = pd.concat([features_scaled_t, train_df[target]], axis=1)


# features_v = val_df.drop(columns='Class')
# features_scaled_v = pd.DataFrame(scaler.fit_transform(features_v), columns=features_v.columns)
# val_df_scaled = pd.concat([features_scaled_v, val_df[target]], axis=1)

# Guardar como CSV
train_df_scaled.to_csv(os.path.join(out_filepath, 'train.csv'), index=False)
val_df.to_csv(os.path.join(out_filepath, 'val.csv'), index=False)
test_df.to_csv(os.path.join(out_filepath, 'test.csv'), index=False)

# Mostrar tamaños relativos
total_size = df_raw.size
print("Tamaño total del DataFrame original:", total_size)
print("Tamaño relativo del conjunto de entrenamiento:", train_df.size / total_size)
print("Tamaño relativo del conjunto de validación:", val_df.size / total_size)
print("Tamaño relativo del conjunto de prueba:", test_df.size / total_size)

# Función para crear ZIPs
def create_zip(source_path, source_file, target_path, target_file):
    source_filepath = os.path.join(source_path, source_file)
    target_filepath = os.path.join(target_path, target_file)
    with ZipFile(target_filepath, 'w') as new_zipfile:
        new_zipfile.write(source_filepath) 

# Crear archivos ZIP
create_zip(out_filepath, 'train.csv', out_filepath, 'train.zip')
create_zip(out_filepath, 'test.csv', out_filepath, 'test.zip')
create_zip(out_filepath, 'val.csv', out_filepath, 'val.zip')

'''
Tamaño total del DataFrame original: 8829017
Tamaño relativo del conjunto de entrenamiento: 0.699968750768064
Tamaño relativo del conjunto de validación: 0.15002791363976306 
Tamaño relativo del conjunto de prueba: 0.15000333559217294    
'''