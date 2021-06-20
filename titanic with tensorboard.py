import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import layers
import numpy as np
import datetime


def titanic_preprocess(df, train=True):
    df = df.set_index('PassengerId')
    del df['Ticket']
    df['Name'] = df['Name'].str.extract(r'([A-Z][a-z]+\.)')

    def numeric_titles(title):
        if title == 'Mr.':
            return 0
        if title == 'Miss.' or title == 'Ms.' or title == 'Mlle.':
            return 1
        if title == 'Mrs.' or title == 'Mme.':
            return 2
        if title == 'Master':
            return 3
        else:
            return 4

    df['Name'] = df['Name'].apply(numeric_titles)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({np.nan: 0, 'S': 1, 'C': 2, 'Q': 3})
    df['Cabin'] = df['Cabin'].apply(lambda x: x[0] if type(x) == str else x)
    df['Cabin'] = df['Cabin'].map({np.nan: 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8})
    conditions = [df['Pclass'] == 1, df['Pclass'] == 2, df['Pclass'] == 3]
    values = [df.groupby('Pclass').mean()['Age'][1], df.groupby('Pclass').mean()['Age'][2],
              df.groupby('Pclass').mean()['Age'][3]]
    df['Age'] = np.where(df['Age'].isnull(), np.select(conditions, values), df['Age'])
    means = df.mean()
    fill_values = {'Pclass': means['Pclass'],
                   'Sex': means['Sex'],
                   'SibSp': means['SibSp'],
                   'Parch': means['Parch'],
                   'Fare': means['Fare']}
    df = df.fillna(value=fill_values)
    if train:
        Y = df.values[:, 0]
        X = df.values[:, 1:]
        X = MinMaxScaler().fit_transform(X)
        return X, Y
    else:
        X = df.values
        X = MinMaxScaler().fit_transform(X)
        return X


def create_model():
    model = keras.Sequential()
    model.add(layers.Dense(32, activation="relu", input_shape=(9,)))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(8, activation="relu"))
    model.add(layers.Dense(8, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    X, Y = titanic_preprocess(df)
    X_test = titanic_preprocess(test_df, train=False)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.shuffle(len(X_train)).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    val_dataset = val_dataset.shuffle(len(X_val)).batch(32)
    model = create_model()
    checkpoint_filepath = 'titanic_net'
    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', save_best_only=True)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(train_dataset, epochs=1000, validation_data=val_dataset, callbacks=[tensorboard_callback, checkpoint])
    model = load_model(checkpoint_filepath)
    Y_pred = model.predict(X_test)
    Y_pred = np.round(Y_pred).squeeze()
    indices = np.array(range(892, 892 + len(Y_pred)))
    d = {'PassengerId': indices, 'Survived': Y_pred}
    submission = pd.DataFrame(d, columns=['PassengerId', 'Survived'])
    submission['Survived'] = submission['Survived'].astype(int)
    submission = submission.set_index('PassengerId')
    submission.to_csv('titanic_submission.csv')

