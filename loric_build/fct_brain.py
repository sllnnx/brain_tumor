from loric_build.nn_and_metrics import *
from PIL import Image
import pandas as pd
import numpy as np
from numpy import asarray


def dim_effect(df_nbs, df_dim, min_dim, dprf):

    percent = np.arange(min_dim, 1, 0.1)
    outputs = []

    for per in percent:
        m_width, m_height = per * df_dim.describe().mean()

        pic_dim = Image.open('dataset/yes/Y1.jpg').convert('L')
        pic_dim = pic_dim.resize((int(m_width), int(m_height)), Image.LANCZOS)
        pic_vec = asarray(pic_dim).ravel()
        dim = len(pic_vec)

        df = pd.DataFrame(columns=range(dim), index=range(len(diagnostic)))
        index = 0

        for nb, name in enumerate(df_nbs.Path):
            for leng in range(df_nbs.Length[nb]):
                filename = name + '/' + os.listdir(name)[leng]
                pic = Image.open(filename).convert('L')
                pic = pic.resize((int(m_width), int(m_height)), Image.LANCZOS)
                pic_vector = asarray(pic).ravel()
                df.iloc[index] = pic_vector
                index += 1

        diag_num = [0 if i == 'no' else 1 if i == "yes" else i for i in diagnostic]
        df['diagnostic'] = diag_num

        X = df.copy()
        X = X.astype('int')
        y = X.pop('diagnostic')
        X = X / 255

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=0)

        y_train = y_train.values.reshape((y_train.shape[0], 1))
        y_test = y_test.values.reshape((y_test.shape[0], 1))

        X_train = X_train.T
        y_train = y_train.T
        X_test = X_test.T
        y_test = y_test.T

        param = L_layer_model(X_train, y_train, layers_dims=[dim, 5, 1], learning_rate=0.0075, num_iterations=2500,
                              print_cost=True)
        pro = predict(X_test, y_test, param)

        outputs.append(pro)

    return outputs