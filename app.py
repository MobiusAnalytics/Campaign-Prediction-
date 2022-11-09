#%%writefile app.py

import pickle
import numpy as np
import streamlit as st
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# loading the trained model
pickle_file = open('Campaign_Prediction_model.sav', 'rb') 
Model = pickle.load(pickle_file)


@st.cache()


# defining the function which will make the prediction using the data which the user inputs 
def prediction(Education,Marital_Status,age,Income,Childern_count,NumWebVisitsMonth,Complain,RFM_Score,Campaign_1_Status,Campaign_2_Status,Campaign_3_Status,Campaign_4_Status,Campaign_5_Status): 
    if Education == "Graduation":
        Education = 0
    elif Education == "Master":
        Education = 1
    elif Education == "Others":
        Education = 2
    elif Education == "PhD":
        Education = 3
        
    if Marital_Status == "Single":
        Marital_Status = 0
    elif Marital_Status == "Married":
        Marital_Status = 1
    elif Marital_Status == "Divorsed":
        Marital_Status = 2
    elif Marital_Status == "Others":
        Marital_Status = 3   
    elif Marital_Status == "Widow":
        Marital_Status = 4   
    
    if Complain == "YES":
        Complain = 1
    elif Complain == "NO":
        Complain = 0
        
    if Campaign_1_Status == "YES":
        Campaign_1_Status = 1
    elif Campaign_1_Status == "NO":
        Campaign_1_Status = 0
        
    if Campaign_2_Status == "YES":
        Campaign_2_Status = 1
    elif Campaign_2_Status == "NO":
        Campaign_2_Status = 0    
    
    if Campaign_3_Status == "YES":
        Campaign_3_Status = 1
    elif Campaign_3_Status == "NO":
        Campaign_3_Status = 0 
    
    if Campaign_4_Status == "YES":
        Campaign_4_Status = 1
    elif Campaign_4_Status == "NO":
        Campaign_4_Status = 0 
    
    if Campaign_5_Status == "YES":
        Campaign_5_Status = 1
    elif Campaign_5_Status == "NO":
        Campaign_5_Status = 0 

    Predict = Model.predict([[Education,Marital_Status,age,Income,Childern_count,NumWebVisitsMonth,Complain,RFM_Score,Campaign_1_Status,Campaign_2_Status,Campaign_3_Status,Campaign_4_Status,Campaign_5_Status]])
    
    if Predict ==1:
        Response = "Based on the given attributes, Model predicts that the Customer may accept the offer in the new Campaign"
    else:
        Response = "Based on the given attributes, Model predicts that the Customer may not accept the offer in the new Campaign"
     
    return Predict
    
       
# this is the main function in which we define our webpage  
def main(): 
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:grey;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Campaign Performance Prediction</h1> 
    </div> 
    """
      
    # display the front end aspect
    
    #st.markdown(html_temp, unsafe_allow_html = True) 
    st.sidebar.title("Campaign Performance Prediction")
    st.sidebar.markdown("- Based on the Customer attributes, Engagement metrics and the previous campaign history the ML Model predicts the performance of the upcoming campaign.")
    st.sidebar.markdown("- The Prediction results helps the company to maximize the profit for the upcoming marketing campaign")
    st.image("""title.png""")
    st.subheader("Upload a file to Predict the output!")
    uploaded_file = st.file_uploader("Choose a File")
    if uploaded_file is not None:
    # To predict a test dataframe!!!
        dataframe = pd.read_csv(uploaded_file)
        dataframe_org = dataframe.copy()
        dataframe = dataframe.replace({'Education' : { 'Graduation' : 0, 'Master' : 1, 'PhD' : 3,'Others':2}})
        dataframe = dataframe.replace({'Marital_Status' : { 'Single' : 0, 'Married' : 1, 'Divorced' : 2,'Widow':4,'Others':3}})
        output = Model.predict(dataframe)
        #output = int(output)            
        dataframe_org['Response'] = output
        dataframe_org = dataframe_org.replace({'Response' : { 0 :'Reject', 1:'Accept'}})
        st.write(dataframe_org)
        dataframe_org =dataframe_org.to_csv(index=False).encode('utf-8')
        st.download_button(label='Download CSV',data=dataframe_org,mime='text/csv',file_name='Download.csv')
    
    # following lines create boxes in which user can enter data required to make prediction 
    st.subheader("")
    st.subheader("Customer attributes")
    Education = st.selectbox('Educational Qualification',("Graduation","Master","PhD","Others"))
    Marital_Status = st.selectbox('Marital Status',("Single","Married","Divorced","Widow","Others"))
    age = st.number_input("Age of the Customer",min_value=18,max_value=80,step=1)
    Income = st.number_input("Income of the Customer",min_value=1000,max_value=1000000)
    Childern_count = st.number_input("No of Children in the Family")
    st.subheader("Engagement Metrics")
    NumWebVisitsMonth = st.number_input("No Web Visits made within a month",min_value=0,max_value=25,step=1)
    Complain = st.selectbox('Any Complaints from Customer',("YES","NO"))
    RFM_Score = st.number_input("Recency-Frequency-Monetary Score",min_value=0.10,max_value=5.00)
    st.subheader("Previous Campaign Status")
    Campaign_1_Status = st.selectbox('Did the Customer accepted Campaign 1 ?',("YES","NO"))
    Campaign_2_Status = st.selectbox('Did the Customer accepted Campaign 2 ?',("YES","NO"))
    Campaign_3_Status = st.selectbox('Did the Customer accepted Campaign 3 ?',("YES","NO"))
    Campaign_4_Status = st.selectbox('Did the Customer accepted Campaign 4 ?',("YES","NO"))
    Campaign_5_Status = st.selectbox('Did the Customer accepted Campaign 5 ?',("YES","NO"))
    
    result = ""
    
    st.markdown("Predict the given customer's response on the planned campaign")
    
    if st.button("PREDICT"): 
        result = prediction(Education,Marital_Status,age,Income,Childern_count,NumWebVisitsMonth,Complain,RFM_Score,Campaign_1_Status,Campaign_2_Status,Campaign_3_Status,Campaign_4_Status,Campaign_5_Status)
        if result == 0:
            st.image("""fail.png""")
            st.warning("Based on the given attributes, Model predicts that the Customer may not accept the offer in the new Campaign", icon="⚠️")
        else:
            st.image("""success.png""")
            st.success('Based on the given attributes, Model predicts that the Customer will accept the offer in the new Campaign',icon="✅")
    
if __name__=='__main__': 
    main()