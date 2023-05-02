import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Iris Flower Dataset", page_icon="📈")

st.write("""
# Simple Iris Flower Prediction App

¡Esta aplicación predice **el tipo de flor de Iris**!
""")

st.sidebar.header('Parámetros de entrada del usuario')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Parámetros de entrada de usuario')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Etiquetas de clase y su número de índice correspondiente')
st.write(iris.target_names)

st.subheader('Predicción')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Probabilidad de predicción')
st.write(prediction_proba)

