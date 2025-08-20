from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)


model = pickle.load(open('model/model_01.pkl', 'rb'))

def preprocess_input_data(input_data):
    """
    Apply one-hot encoding correctly and select first 26 features
    """
    data = input_data.copy()
    
    print("Original data before encoding:")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Data:\n{data}")
    
    
    categorical_columns = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
    
    
    data_encoded = pd.get_dummies(
        data, 
        columns=categorical_columns, 
        drop_first=False,
        dummy_na=False
    )
    
    print("After one-hot encoding:")
    print(f"Shape: {data_encoded.shape}")
    print(f"Total columns: {len(data_encoded.columns)}")
    print(f"Column names: {list(data_encoded.columns)}")
    
    
    if data_encoded.shape[1] < 26:
        print(f"ERROR: Only {data_encoded.shape[1]} columns after encoding, need at least 26")
        
        return create_manual_encoding(data)
    
    
    final_data = data_encoded.iloc[:, :26]
    
    print(f"Final shape: {final_data.shape}")
    print(f"Final columns: {list(final_data.columns)}")
    
    return final_data

def create_manual_encoding(data):
    """
    Manually create one-hot encoded features if pd.get_dummies fails
    """
    result = data.copy()
    
    
    categorical_columns = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
    
    for col in categorical_columns:
        if col in result.columns:
            result = result.drop(columns=[col])
    
    
    result['TypeofContact_Company Invited'] = 1 if data['TypeofContact'].iloc[0] == 'Company Invited' else 0
    result['TypeofContact_Self Enquiry'] = 1 if data['TypeofContact'].iloc == 'Self Enquiry' else 0
    
    
    result['Occupation_Free Lancer'] = 1 if data['Occupation'].iloc == 'Free Lancer' else 0
    result['Occupation_Large Business'] = 1 if data['Occupation'].iloc == 'Large Business' else 0
    result['Occupation_Salaried'] = 1 if data['Occupation'].iloc == 'Salaried' else 0
    result['Occupation_Small Business'] = 1 if data['Occupation'].iloc == 'Small Business' else 0
    
    
    result['Gender_Female'] = 1 if data['Gender'].iloc == 'Female' else 0
    result['Gender_Male'] = 1 if data['Gender'].iloc == 'Male' else 0
    
    
    result['ProductPitched_Basic'] = 1 if data['ProductPitched'].iloc == 'Basic' else 0
    result['ProductPitched_Deluxe'] = 1 if data['ProductPitched'].iloc == 'Deluxe' else 0
    result['ProductPitched_King'] = 1 if data['ProductPitched'].iloc == 'King' else 0
    result['ProductPitched_Standard'] = 1 if data['ProductPitched'].iloc == 'Standard' else 0
    result['ProductPitched_Super Deluxe'] = 1 if data['ProductPitched'].iloc == 'Super Deluxe' else 0
    
    
    result['MaritalStatus_Divorced'] = 1 if data['MaritalStatus'].iloc == 'Divorced' else 0
    result['MaritalStatus_Married'] = 1 if data['MaritalStatus'].iloc == 'Married' else 0
    result['MaritalStatus_Unmarried'] = 1 if data['MaritalStatus'].iloc == 'Unmarried' else 0
    
    
    result['Designation_AVP'] = 1 if data['Designation'].iloc == 'AVP' else 0
    result['Designation_Executive'] = 1 if data['Designation'].iloc == 'Executive' else 0
    result['Designation_Manager'] = 1 if data['Designation'].iloc == 'Manager' else 0
    result['Designation_Senior Manager'] = 1 if data['Designation'].iloc == 'Senior Manager' else 0
    result['Designation_VP'] = 1 if data['Designation'].iloc == 'VP' else 0
    
    print(f"Manual encoding result shape: {result.shape}")
    print(f"Manual encoding columns: {list(result.columns)}")
    
    
    final_result = result.iloc[:, :26]
    
    return final_result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_form')
def predict_form():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            input_data = pd.DataFrame({
                'Age': [float(request.form['age'])],
                'TypeofContact': [request.form['type_of_contact']],
                'CityTier': [int(request.form['city_tier'])],
                'DurationOfPitch': [float(request.form['duration_of_pitch'])],
                'Occupation': [request.form['occupation']],
                'Gender': [request.form['gender']],
                'NumberOfFollowups': [float(request.form['number_of_followups'])],
                'ProductPitched': [request.form['product_pitched']],
                'PreferredPropertyStar': [float(request.form['preferred_property_star'])],
                'MaritalStatus': [request.form['marital_status']],
                'NumberOfTrips': [float(request.form['number_of_trips'])],
                'Passport': [int(request.form['passport'])],
                'PitchSatisfactionScore': [int(request.form['pitch_satisfaction_score'])],
                'OwnCar': [int(request.form['own_car'])],
                'Designation': [request.form['designation']],
                'MonthlyIncome': [float(request.form['monthly_income'])],
                'TotalVisiting': [float(request.form['total_visiting'])]
            })
            
            print("="*50)
            print("STARTING PREDICTION PROCESS")
            print("="*50)
            
            
            processed_data = preprocess_input_data(input_data)
            
            print("="*50)
            print("MAKING PREDICTION")
            print(f"Final processed data shape: {processed_data.shape}")
            print("="*50)
            
            
            prediction = model.predict(processed_data)
            print(f"Prediction result: {prediction}")
            
            
            try:
                prediction_proba = model.predict_proba(processed_data)
                probability = round(prediction_proba[0][1] * 100, 2)
                print(f"Prediction probability: {probability}%")
            except Exception as prob_error:
                print(f"Probability calculation error: {prob_error}")
                probability = None
            
            
            if prediction[0] == 1:
                result = "Yes! Customer WILL TAKE the travel product!"
                result_style = "success"
            else:
                result = "No, Customer will NOT take the travel product"
                result_style = "warning"
            
            print(f"Final result: {result}")
            
            return render_template('home.html', 
                                 prediction_result=result,
                                 probability=probability,
                                 result_style=result_style)
        
        except Exception as e:
            error_message = f"Error making prediction: {str(e)}"
            print("="*50)
            print("ERROR OCCURRED")
            print(f"Error: {error_message}")
            print(f"Error type: {type(e).__name__}")
            print("="*50)
            return render_template('home.html', 
                                 prediction_result=error_message,
                                 probability=None)

if __name__ == '__main__':
    app.run(debug=True)
