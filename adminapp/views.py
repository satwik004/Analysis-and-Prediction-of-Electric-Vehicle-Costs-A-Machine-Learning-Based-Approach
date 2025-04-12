from django.shortcuts import render,redirect
from mainapp.models import*
from userapp.models import*
from adminapp.models import *
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import HttpResponse
import os
import shutil
from django.shortcuts import render
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from django.contrib import messages
from .models import LR,RF,ANN
from keras.models import load_model


#gradient boost machine algo for getting acc ,precession , recall , f1 score
# Create your views here.
def adminlogout(req):
    messages.info(req,'You are logged out...!')
    return redirect('admin')
def admindashboard(req):
    all_users_count =  User.objects.all().count()
    pending_users_count = User.objects.filter(User_Status = 'Pending').count()
    rejected_users_count = User.objects.filter(User_Status = 'removed').count()
    accepted_users_count =User.objects.filter(User_Status = 'accepted').count()
    Feedbacks_users_count= Feedback.objects.all().count()
    user_uploaded_images =Dataset.objects.all().count()
    return render(req,'admin/admin-dashboard.html',{'a' : all_users_count, 'b' : pending_users_count, 'c' : rejected_users_count, 'd' : accepted_users_count, 'e':Feedbacks_users_count, 'f':user_uploaded_images})

def pendingusers(req):
    pending = User.objects.filter(User_Status = 'Pending')
    paginator = Paginator(pending, 5) 
    page_number = req.GET.get('page')
    post = paginator.get_page(page_number)
    return render(req,'admin/admin-pending-users.html', { 'user' : post})

def delete_user(req, id):
    User.objects.get(User_id = id).delete()
    messages.warning(req, 'User was Deleted..!')
    return redirect('manageusers')

def accept_user(req, id):
    status_update = User.objects.get(User_id = id)
    status_update.User_Status = 'accepted'
    status_update.save()
    messages.success(req, 'User was accepted..!')
    return redirect('pendingusers')

def manageusers(req):
    manage_users  = User.objects.all()
    paginator = Paginator(manage_users, 5)
    page_number = req.GET.get('page')
    post = paginator.get_page(page_number)
    return render(req, 'admin/admin-manage-users.html', {"allu" : manage_users, 'user' : post})

def reject_user(req, id):
    status_update2 = User.objects.get(User_id = id)
    status_update2.User_Status = 'removed'
    status_update2.save()
    messages.warning(req, 'User was Rejected..!')
    return redirect('pendingusers')

def admin_datasetupload(req):
    return render(req,'admin/admin-upload-dataset.html')
def admin_dataset_btn(req):
    messages.success(req, 'Dataset uploaded successfully..!')
    return redirect('admin_datasetupload')







def adminfeedback(req):
    feed =Feedback.objects.all()
    return render(req,'admin/user-feedback.html', {'back':feed})

def adminsentiment(req):
    fee = Feedback.objects.all()
    return render(req,'admin/user-sentiment.html' , {'cat':fee})

def usergraph(req):
    positive = Feedback.objects.filter(Sentiment = 'positive').count()
    very_positive = Feedback.objects.filter(Sentiment = 'very positive').count()
    negative = Feedback.objects.filter(Sentiment = 'negative').count()
    very_negative = Feedback.objects.filter(Sentiment = 'very negative').count()
    neutral = Feedback.objects.filter(Sentiment = 'neutral').count()
    context ={
        'vp': very_positive, 'p':positive, 'n':negative, 'vn':very_negative, 'ne':neutral
    }
    return render(req,'admin/user-sentiment-graph.html',context)

def LR_alg(req):
  return render(req,'admin/LR_alg.html')

def RF_alg(req):
  return render(req,'admin/RF_alg.html')

def ANN_alg(req):
  return render(req,'admin/ANN_alg.html')



def LR_btn(req):
    import time
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    

    # Sample dataset
    data = pd.read_csv('Dataset/ElectricCarData_Clean3.csv') 
    

    df = pd.DataFrame(data)


    #  # Check the columns in the dataset
    print(df.columns)

    # One-Hot Encoding for categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)  # Converts categorical variables to numeric

    # Features and target variable
    X = df_encoded.drop('Price', axis=1)  # Use the encoded DataFrame
    y = df_encoded['Price']

    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Measure training time
    start_time = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    # Prediction time
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    prediction_time = end_time - start_time

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_test, y_pred)

    # Accuracy: Percentage of predictions within 10% of actual values
    accuracy_threshold = 0.1  # 10%
    accuracy = np.mean(np.abs((y_test - y_pred) / y_test) <= accuracy_threshold) * 100

    # Cross-validation for overfitting check
    cv_scores = cross_val_score(model, X_train, y_train, cv=2, scoring='r2')


    print(f"TrainingTime: {training_time} seconds")
    print(f"PredictionTime: {prediction_time} seconds")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r_squared}")
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Model Score (R²): {model.score(X_test, y_test)}")
    print(f"Accuracy (within 10% threshold): {accuracy}%")

    # Prepare the results to pass to the template
    data = {
        "Training_Time": training_time,
        "Prediction_Time": prediction_time,
        "RMSE": rmse,
        "R_squared": r_squared,
        "Cross_Validation_Scores":cv_scores,
        "Model_Score_r2":model.score(X_test, y_test),
        "Accuracy": accuracy
    }
    # name = "CNN Text Classification"
    LR.objects.create(accuracy=accuracy, mse=mse, rmse=rmse)

    data1 = LR.objects.last()

    # Save the trained model (optional)
    joblib.dump(model, 'ann_model.pkl')

    return render(req,'admin/LR_alg.html',data)

def RF_btn(req):
    import time
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor  # Random Forest Regressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib

    # Sample dataset
    data = pd.read_csv('Dataset/ElectricCarData_Clean3.csv')
    df = pd.DataFrame(data)

    # Check the columns in the dataset
    print(df.columns)

    # One-Hot Encoding for categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)  # Converts categorical variables to numeric

    # Features and target variable
    X = df_encoded.drop('Price', axis=1)  # Use the encoded DataFrame
    y = df_encoded['Price']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Measure training time
    start_time = time.time()
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # Random Forest Regressor
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    # Prediction time
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    prediction_time = end_time - start_time

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_test, y_pred)

    # Accuracy: Percentage of predictions within 10% of actual values
    accuracy_threshold = 0.1  # 10%
    accuracy = np.mean(np.abs((y_test - y_pred) / y_test) <= accuracy_threshold) * 100

    # Cross-validation for overfitting check
    cv_scores = cross_val_score(model, X_train, y_train, cv=2, scoring='r2')

    print(f"Training Time: {training_time} seconds")
    print(f"Prediction Time: {prediction_time} seconds")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r_squared}")
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Model Score (R²): {model.score(X_test, y_test)}")
    print(f"Accuracy (within 10% threshold): {accuracy}%")

    # Prepare the results to pass to the template
    data = {
        "Training_Time": training_time,
        "Prediction_Time": prediction_time,
        "RMSE": rmse,
        "R_squared": r_squared,
        "Cross_Validation_Scores": cv_scores,
        "Model_Score_r2": model.score(X_test, y_test),
        "Accuracy": accuracy
    }
    RF.objects.create(accuracy=accuracy, mse=mse, rmse=rmse)

    # Save the trained model (optional)
    joblib.dump(model, 'rf1.pkl')

    return render(req, 'admin/RF_alg.html', data)


def ANN_btn(req):
    import time
    import numpy as np
    import pandas as pd
    from sklearn.neural_network import MLPRegressor  # MLPRegressor for ANN
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler  # Required for ANN scaling
    import joblib

    # Sample dataset
    data = pd.read_csv('Dataset/ElectricCarData_Clean3.csv')
    df = pd.DataFrame(data)

    # Check the columns in the dataset
    print(df.columns)

    # One-Hot Encoding for categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)  # Converts categorical variables to numeric

    # Features and target variable
    X = df_encoded.drop('Price', axis=1)  # Use the encoded DataFrame
    y = df_encoded['Price']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ANN requires feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Measure training time
    start_time = time.time()
    model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)  # ANN with two hidden layers
    model.fit(X_train_scaled, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    # Prediction time
    start_time = time.time()
    y_pred = model.predict(X_test_scaled)
    end_time = time.time()
    prediction_time = end_time - start_time

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_test, y_pred)

    # Accuracy: Percentage of predictions within 10% of actual values
    accuracy_threshold = 0.1  # 10%
    accuracy = np.mean(np.abs((y_test - y_pred) / y_test) <= accuracy_threshold) * 100

    # Cross-validation for overfitting check
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=2, scoring='r2')

    print(f"Training Time: {training_time} seconds")
    print(f"Prediction Time: {prediction_time} seconds")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r_squared}")
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Model Score (R²): {model.score(X_test_scaled, y_test)}")
    print(f"Accuracy (within 10% threshold): {accuracy}%")

    # Prepare the results to pass to the template
    data = {
        "Training_Time": training_time,
        "Prediction_Time": prediction_time,
        "RMSE": rmse,
        "R_squared": r_squared,
        "Cross_Validation_Scores": cv_scores,
        "Model_Score_r2": model.score(X_test_scaled, y_test),
        "Accuracy": accuracy
    }
    ANN.objects.create(accuracy=accuracy, mse=mse, rmse=rmse)

    # Save the trained model (optional)
    joblib.dump(model, 'ann_regressor_model.pkl')

    return render(req, 'admin/ANN_alg.html', data)


def admingraph(req):
    details1 = LR.objects.last()
    LR1 = details1.accuracy

    details2 = RF.objects.last()
    RF1 = details2.accuracy

    details3 = ANN.objects.last()
    ANN1 = details3.accuracy

    print('LR1','RF1','ANN1')
    print(LR1,RF1,ANN1)
    return render(req,'admin/admin-graph-analysis.html',{'LR':LR1, 'RF':RF1,'ANN':ANN1})




















