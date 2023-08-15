import pickle
import pandas as pd
import numpy as np

 

#loading the models
diabetes = read_csv(open("diabetes.csv"))
heart_disease = read_csv(open("heart.csv"))
lung_cancer = read_csv(open("lung_cancer.csv"))

 

 

#sidebar for navigation

 

with st.sidebar:

    selected = option_menu("Multiple Disease Prediction System using Machine Learning", 

                           ["Diabetes Prediction",
                            "Heart Disese Prediction",
                            "Kidney Disease Prediction",
                            "Liver Prediction"],

                           icons = ["activity", "heart-fill", "people-fill", 
                                    "gender-female", "apple"],

                           default_index = 0)

 

 

 

 


#Diabetes Prediction Page:
if(selected == "Diabetes Prediction"):

    #page title
    st.title("Diabetes Prediction")



 

# getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")

    with col2:
        Glucose = st.text_input("Glucose Level")

    with col3:
        BloodPressure = st.text_input("Blood Pressure Value")

    with col1:
        SkinThickness = st.text_input("Skin Thickness Value")

    with col2:
        Insulin = st.text_input("Insulin Level")

    with col3:
        BMI = st.text_input("BMI Value")

    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")

    with col2:
        Age = st.text_input("Age of the Person")

 


# code for Prediction
    diabetes_diagnosis = " "

    # creating a button for Prediction

    if st.button("Diabetes Test Result"):
        diabetes_prediction = diabetes.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        if (diabetes_prediction[0] == 0):
          diabetes_diagnosis = "Hurrah! You have no Diabetes."
        else:
          diabetes_diagnosis = "Sorry! You have Diabetes."

    st.success(diabetes_diagnosis)

 

 

 

#Heart Disease Prediction Page:
if(selected == "Heart Disese Prediction"):

    #page title
    st.title("Heart Disease Prediction")



# getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age")

    with col2:
        sex = st.number_input("Sex")

    with col3:
        cp = st.number_input("Chest Pain Types")

    with col1:
        trestbps = st.number_input("Resting Blood Pressure")

    with col2:
        chol = st.number_input("Serum Cholestoral in mg/dl")

    with col3:
        fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl")

    with col1:
        restecg = st.number_input("Resting Electrocardiographic Results")

    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved")

    with col3:
        exang = st.number_input("Exercise Induced Angina")

    with col1:
        oldpeak = st.number_input("ST Depression induced by Exercise")

    with col2:
        slope = st.number_input("Slope of the peak exercise ST Segment")

    with col3:
        ca = st.number_input("Major vessels colored by Flourosopy")

    with col1:
        thal = st.number_input("thal: 0 = normal; 1 = fixed defect; 2 = reversable defect")




    # code for Prediction
    heart_diagnosis = " "

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          

        if (heart_prediction[0] == 0):
          heart_diagnosis = "Hurrah! Your Heart is Good."
        else:
          heart_diagnosis = "Sorry! You have Heart Problem."

    st.success(heart_diagnosis)


 

 

 


#Lung Cancer Prediction Page:
if(selected == "Lung Cancer Prediction"):
    #page title
    st.title("Lung Cancer Prediction")

 

 

# getting the input data from the user
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        GENDER = st.number_input("GENDER")

    with col2:
        AGE = st.number_input("AGE")

    with col3:
        SMOKING = st.number_input("SMOKING")

    with col4:
        YELLOW_FINGERS = st.number_input("YELLOW_FINGERS")

    with col1:
        ANXIETY = st.number_input("ANXIETY")

    with col2:
        PEER_PRESSURE = st.number_input("PEER_PRESSURE")

    with col3:
        CHRONIC_DISEASE = st.number_input("CHRONIC DISEASE")

    with col4:
        FATIGUE = st.number_input("FATIGUE")

    with col1:
        ALLERGY = st.number_input("ALLERGY")

    with col2:
        WHEEZING = st.number_input("WHEEZING")

    with col3:
        ALCOHOL_CONSUMING = st.number_input("ALCOHOL CONSUMING")

    with col4:
        COUGHING = st.number_input("COUGHING")

    with col1:
        SHORTNESS_OF_BREATH = st.number_input("SHORTNESS OF BREATH")

    with col2:
        SWALLOWING_DIFFICULTY = st.number_input("SWALLOWING DIFFICULTY")

    with col3:
        CHEST_PAIN = st.number_input("CHEST PAIN")


 


# code for Prediction
    lung_cancer_result = " "

    # creating a button for Prediction
    if st.button("Lung Cancer Test Result"):
        lung_cancer_report = lung_cancer.predict([[GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]])

        if (lung_cancer_report[0] == 0):
          lung_cancer_result = "Hurrah! You have no Lung Cancer."
        else:
          lung_cancer_result = "Sorry! You have Lung Cancer."

    st.success(lung_cancer_result)
    
