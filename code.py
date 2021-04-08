import numpy as np 
import pandas as pd

import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image as Img

st.sidebar.image("/Users/ewereasaka/Desktop/AI/cover.png")
st.title("""
Stroke Detection """)
st.write("""  Let's find out if you are likely to get Stroke or not !""")

image = Img.open('/Users/ewereasaka/Desktop/AI/brain.png')
st.sidebar.image(image)

st.sidebar.subheader(""" Did you know that stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths ? """)
st.sidebar.write(""" 
Based on the data, gender does not affect the probability of stroke, both the Genders have arround 5% chance.


There is a large correlation between age and stroke risk. However, this can also happen in children.

The presence of hypertension or heart diseases have affect on stroke risk.

People who are married have a higher risk of stroke. 

Working with children in kindergartens and schools or caring for their children have a positive impact on the health of adults. Self Employed people are at a higher risk compared to Private and Government Jobs.


Living in a rural or urban areas does not show much difference.


Smoking also affects the risk of stroke. 


Obesity leads not only to problems with blood vessels and many others, but also to the risk of stroke.

Diabetes has an impact on the risk of stroke.
 """)


stroke = pd.read_csv('Stroke.csv') #Import data set
# Round off Age
stroke['age'] = stroke['age'].apply(lambda x : round(x))

# Sorting DataFrame based on Gender then on Age and using Forward Fill-ffill() to fill NaN value for BMI
stroke.sort_values(['gender', 'age'], inplace=True) 
stroke.reset_index(drop=True, inplace=True)
stroke['bmi'].ffill(inplace=True)


gender_dict = {'Male': 0, 'Female': 1, 'Other': 2}
ever_married_dict = {'No': 0, 'Yes': 1}
work_type_dict = {'children': 0, 'Never_worked': 1, 'Govt_job': 2, 'Private': 3, 'Self-employed': 4}
residence_type_dict = {'Rural': 0, 'Urban': 1}
smoking_status_dict = {'Unknown': 0, 'never smoked': 1, 'formerly smoked':2, 'smokes': 3}

stroke['gender'] = stroke['gender'].map(gender_dict)
stroke['ever_married'] = stroke['ever_married'].map(ever_married_dict)
stroke['work_type'] = stroke['work_type'].map(work_type_dict)
stroke['Residence_type'] = stroke['Residence_type'].map(residence_type_dict)
stroke['smoking_status'] = stroke['smoking_status'].map(smoking_status_dict)
X = stroke.drop(columns=['id', 'stroke'])
y = stroke['stroke']

from imblearn.over_sampling import SMOTE

#Using SMOTE to balance the Data

smote = SMOTE(random_state = 2)
X, y = smote.fit_resample(X, y) 

# Spliting the Data into Train and Test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
Name=st.text_input("Name?")

def get_user_input():
    genders=["Male", "Female"]
    gender=st.radio("Gender?", genders)
    
    age=st.number_input("Your age?")
    option =["Yes", "No"]
    hypertension=st.radio("Do you have Hypertension?", option)
    
    heart_disease=st.radio("Do you have a Heart disease?", option)
    
    
    ever_married=st.radio("Have you ever married?", option)
   
    work_type = st.selectbox('Select Work Type',('Private', "Self-employed", "Government Job", "Children", "Never_worked"))
    Residence_type = st.selectbox('Select Residency Type',('Urban', "Rural"))
    st.text("Let's alculate your BMI for you by inputing weight and height")
    height=st.number_input("Your height in metres?", min_value=0.1)
    weight=st.number_input("Your weight in kilograms?",min_value=1)
    bmi=weight/(height*height)
    st.text("Or input it directly")
    bmi=st.slider('BMI', 12.0, 60.0, 12.0)

    avg_glucose_level=st.slider('Glucose level', 0.0, 250.0, 0.0)
    smoking_status = st.selectbox('Smoking Status',('Formerly Smoked', "Smokes", "Never Smoked"))
    gender_dict = {'Male': 0, 'Female': 1, 'Other': 2}
    ever_married_dict = {'No': 0, 'Yes': 1}
    work_type_dict = {'children': 0, 'Never_worked': 1, 'Govt_job': 2, 'Private': 3, 'Self-employed': 4}
    residence_type_dict = {'Rural': 0, 'Urban': 1}
    smoking_status_dict = {'Unknown': 0, 'never smoked': 1, 'formerly smoked':2, 'smokes': 3}
    if gender== "Male":
        gender=0
    else:
        gender=1
    
    if hypertension == "Yes":
        hypertension=1
    else:
        hypertension=0
    if heart_disease == "Yes":
        heart_disease=1
    else:
        heart_disease=0
    if ever_married == "Yes":
        ever_married=1
    else:
        ever_married=0
    if work_type=="children":
        work_type=0
    elif work_type=="Never_worked":
        work_type=1
    elif work_type=="Govt_job":
        work_type=2
    elif work_type=="Private":
        work_type=3
    else:
        work_type=4
    if Residence_type=="Rural":
        Residence_type=0
    else:
        Residence_type=1
    if smoking_status=="Unknown":
        smoking_status=0
    elif smoking_status=="never smoked":
        smoking_status=1
    else:
        smoking_status=2


    
    user_data = {'gender': gender,
              'age': age,
                 'hypertension': hypertension,
                 'heart_disease': heart_disease,
                 'ever_married': ever_married,
              'work_type': work_type,
              'Residence_type': Residence_type,
                 'bmi': bmi,
                 'avg_glucose_level': avg_glucose_level,'smoking_status': smoking_status}
    features = pd.DataFrame(user_data, index=[0])
    return features


  
user_input = get_user_input()
bt = st.button('Get Result')
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred = rf.predict(user_input)

if bt:
    if pred == 1:
        st.info(" You are likely to have Stroke. Please visit the doctor as soon as possible.")
    if pred == 0:
        st.info( 'You are not likely to having stroke.')
        st.balloons()
st.subheader('Our Accuracy Score')
st.write( str(accuracy_score(y_test, rf.predict(X_test)) * 100) + '%' )
