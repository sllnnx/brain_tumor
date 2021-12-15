import pandas as pd
from PIL import Image
import os
from numpy import asarray
from sklearn.model_selection import train_test_split
from Deep_Neural_Network import *



# Path to folders with pictures

path_yes = "dataset/yes"
path_no = "dataset/no"

nb_yes = len(os.listdir(path_yes))
nb_no = len(os.listdir(path_no))

df_nbs = pd.DataFrame({'Path': [path_yes, path_no], 'Length': [nb_yes, nb_no]})

# Extract pictures dimensions in DF for yes/no files

width = []
height = []
diagnostic = []

for nb, name in enumerate(df_nbs.Path):
    for leng in range(df_nbs.Length[nb]):
        filename = name + '/' + os.listdir(name)[leng]
        pic = Image.open(filename)
        w, h = pic.size
        width.append(w)
        height.append(h)
        diagnostic.append(str.partition(name,'/')[2])

df_dim = pd.DataFrame({'width':width, 'height':height, 'diagnostic':diagnostic})

# Identify min resolution

m_width, m_height = df_dim.describe().min()

# Reshape pictures to normalized dimensions to create DF of identical dimensions
# ? Interpolation type (‘nearest’, ‘lanczos’, ‘bilinear’, ‘bicubic’ or ‘cubic’),
# ? keep ratio?, could try first go then keep ratio if improv metric
# Form DF

df = pd.DataFrame(columns=range(25200), index=range(len(diagnostic)))
index = 0

for nb, name in enumerate(df_nbs.Path):
    for leng in range(df_nbs.Length[nb]):
        filename = name + '/' + os.listdir(name)[leng]
        pic = Image.open(filename).convert('L')
        pic = pic.resize((int(m_width), int(m_height)), Image.NEAREST)
        pic_vector = asarray(pic).ravel()
        df.iloc[index] = pic_vector
        index += 1

diag_num = [0 if i == 'no' else 1 if i == "yes" else i for i in diagnostic]
df['diagnostic'] = diag_num

#SPlit data

X = df.copy()
y = X.pop('diagnostic')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)


y_train = y_train.values.reshape((y_train.shape[0], 1))
y_test = y_test.values.reshape((y_test.shape[0], 1))

