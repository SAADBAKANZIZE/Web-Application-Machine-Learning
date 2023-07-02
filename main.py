import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write('''
# It is a web application that predicts the type of IRIS flower based on the information provided by the user
THIS APP PREDICT THE TYPE OF THE IRIS FLOWER
''')

st.sidebar.header("INPUT PARAMETERS")

def user_input():
    sepal_length=st.sidebar.slider('The length Sepal',4.3,7.9,5.3)
    sepal_width=st.sidebar.slider('The width of Sepal',2.0,4.4,3.3)
    petal_length=st.sidebar.slider('The length of Petal',1.0,6.9,2.3)
    petal_width=st.sidebar.slider('The width of Petal',0.1,2.5,1.3)
    data={'sepal_length':sepal_length,
    'sepal_width':sepal_width,
    'petal_length':petal_length,
    'petal_width':petal_width
    }
    fleur_parametres=pd.DataFrame(data,index=[0])
    return fleur_parametres

df=user_input()

st.subheader('THE GOAL IS TO KNOW THE TYPE OF THIS IRIS')
st.write(df)

iris=datasets.load_iris()
clf=RandomForestClassifier()
clf.fit(iris.data,iris.target)

prediction=clf.predict(df)

st.subheader("THE TYPE OF THIS IRIS FLOWER IS:")
st.write(iris.target_names[prediction])
