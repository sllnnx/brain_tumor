import pandas as pd
from PIL import Image
import os
from numpy import asarray
from sklearn.model_selection import train_test_split
from loric_build.nn_and_metrics import *

# Path to folders with pictures

os.chdir('C:/Users/lpetr/PycharmProjects/brain_tumor')

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

m_width, m_height = 0.3*df_dim.describe().min()

# Reshape pictures to normalized dimensions to create DF of identical dimensions
# ? Interpolation type (‘nearest’, ‘lanczos’, ‘bilinear’, ‘bicubic’ or ‘cubic’),
# ? keep ratio?, could try first go then keep ratio if improv metric
# Form DF

pic_dim = Image.open('dataset/yes/Y1.jpg').convert('L')
pic_dim = pic_dim.resize((int(m_width),int(m_height)), Image.NEAREST)
pic_vec = asarray(pic_dim).ravel()
dim = len(pic_vec)

df = pd.DataFrame(columns=range(dim), index=range(len(diagnostic)))
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
X = X.astype('float64')
y = X.pop('diagnostic')
X = X/255

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, shuffle = True, random_state=0)

y_train = y_train.values.reshape((y_train.shape[0], 1))
y_test = y_test.values.reshape((y_test.shape[0], 1))

X_train = X_train.T
y_train = y_train.T
X_test = X_test.T
y_test = y_test.T

param = L_layer_model(X_train, y_train, layers_dims = [dim, 5, 1], learning_rate = 0.01, num_iterations = 6000, print_cost = True)
pro = predict(X_test, y_test, param)
