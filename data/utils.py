import dill
import numpy as np
from numpy.random import choice
import torch
from torch.utils.data import Dataset

import numpy as np
from librosa import load, amplitude_to_db
from librosa.feature import melspectrogram


def get_spectrogram(path):
    y, sr = load(path)
    S = melspectrogram(y, sr=sr, n_mels=100)
    log_S = amplitude_to_db(S, ref_power=np.max)
    return log_S

class VoicesData:
    def __init__(self, path='users.dl'):
        # Путь до dill модели
        self.path = path
        # Defaultdict информации
        self.base = dill.load(open(self.path, 'rb'))

    def __getitem__(self, item):
        return self.base[item]

    def save(self):
        """Поскольку это не стандартная DB, тут нужна функция сохранения"""
        dill.dump(self.base, open(self.path, 'wb'))

    def get_train(self, shape=(100, 20)):
        train = []
        for key in self.base.keys():
            if len(self.base[key]) >= 2:
                for _ in range(0, len(self.base[key]), 2):
                    values = choice(list(self.base[key]), 2)
                    
                    a = np.asarray(self.base[key][values[0]])
                    b = np.asarray(self.base[key][values[1]])

                    other = choice(list(self.base), 1)[0]
                    c = np.asarray(self.base[other][choice(list(self.base[other]), 1)[0]])

                    other_value = c.copy()
                    value_first = a.copy()
                    value_second = b.copy()

                    other_value.resize(shape)
                    value_first.resize(shape)
                    value_second.resize(shape)

                    train.append((value_first, other_value, value_second))

        return train

    def get_train_for_user(self, user, shape=(100, 20)):
        """Трейн данные для одного пользователя"""
        train = []
        if len(self.base[user]) >= 2:
            for _ in range(0, len(self.base[user])):
                values = choice(list(self.base[user]), 2)

                a = np.asarray(self.base[user][values[0]])
                b = np.asarray(self.base[user][values[1]])

                other = choice(list(self.base), 1)[0]
                c = np.asarray(self.base[other][choice(list(self.base[other]), 1)[0]])

                value_first = a.copy()
                value_second = b.copy()
                other_value = c.copy()

                other_value.resize(shape)
                value_first.resize(shape)
                value_second.resize(shape)

                train.append((value_first, other_value, value_second))
        else:
            raise ValueError
        return train

class VoiceDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(np.array(X)).float()
        
    def __len__(self):
        return self.X.size()[0]
    
    def __getitem__(self, index):
        return self.X[index]