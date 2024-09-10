import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def load_and_preporcess_data(filepath):
    data = pd.read_csv(file)
    X = data.drop(columns=['over_under'])
    y = data['over_under']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def build_model(input_dim):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, input_dim=input_dim, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train_model(model, X_train, y_train):
    history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)
    return history

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy}')
    return accuracy

def save_model(model, filepath):
    model.save(filepath)

def load_model(filepath):
    return tf.keras.models.load_model(filepath)
