# python3 main.py train.tsv test.tsv

import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr


def ATM_Regression(train_data,test_data):

#   set some standard, high scores mean greater for model practice
    replace_dict = {
        "Bigger Towns": 100,
        "Semi Urban": 75,
        "Town": 50,
        "Urban": 25,
        'Checkdrop and Withdraw': 100, 
        'Deposit and Withdraw': 75,
        'Passbook Printing and Withdraw': 50,
        'Only WIthdraw': 25, 
        'Festival':100,
        'National Holiday':75,
        'Strike':50,
        'Working':25,
        'New':100,
        'Normal':70,
        'Old and Dull':40,
        'Facing Road':100,
        'Little Inside':50,
        'Building':100,
        'Flat':85,
        'House':70,
        'Petrol Bunk':55,
        'Shed':40,
        'Shop':25,
        'C':100,'FV':80,'RH':60,'RL':40,'RM':20
    }


    train_data["ATM_TYPE"] = train_data["ATM_TYPE"].replace(replace_dict)
    train_data["ATM_Location_TYPE"]= train_data["ATM_Location_TYPE"].replace(replace_dict)
    train_data["Day_Type"]=train_data["Day_Type"].replace(replace_dict)
    train_data["ATM_looks"]=train_data["ATM_looks"].replace(replace_dict)
    train_data["ATM_Placement"]=train_data["ATM_Placement"].replace(replace_dict)
    train_data["ATM_Attached_to"]=train_data["ATM_Attached_to"].replace(replace_dict)
    train_data["ATM_Zone"]=train_data["ATM_Zone"].replace(replace_dict)

    test_data["ATM_TYPE"] = test_data["ATM_TYPE"].replace(replace_dict)
    test_data["ATM_Location_TYPE"]= test_data["ATM_Location_TYPE"].replace(replace_dict)
    test_data["Day_Type"]=test_data["Day_Type"].replace(replace_dict)
    test_data["ATM_looks"]=test_data["ATM_looks"].replace(replace_dict)
    test_data["ATM_Placement"]=test_data["ATM_Placement"].replace(replace_dict)
    test_data["ATM_Attached_to"]=test_data["ATM_Attached_to"].replace(replace_dict)
    test_data["ATM_Zone"]=test_data["ATM_Zone"].replace(replace_dict)

    # Remove any rows with missing values
    train_data = train_data.dropna()
    test_data = test_data.dropna()

    train_data['Potential_Customers_house'] = train_data['No_of_Other_ATMs_in_1_KM_radius']*train_data['Estimated_Number_of_Houses_in_1_KM_Radius']*train_data['ATM_Zone']
    train_data['ATM_Place_Attached_link_rating']=train_data["ATM_Attached_to"]*train_data['rating']*train_data['No_of_Other_ATMs_in_1_KM_radius']
    train_data['ATM_look_link_rating']=train_data["ATM_looks"]*train_data['rating']
    train_data['ATM_Type_link_rating']=train_data["ATM_TYPE"]*train_data["ATM_Placement"]*train_data['rating']
    train_data['ATM_More_choice_link_look_type']=train_data['No_of_Other_ATMs_in_1_KM_radius']+(train_data["ATM_TYPE"]*train_data['ATM_look_link_rating'])
    train_data['Wait_Time_connect_ATM_Zone']=train_data['Average_Wait_Time']*train_data['No_of_Other_ATMs_in_1_KM_radius']*train_data['ATM_Zone']*train_data['rating']

    test_data['Potential_Customers_house'] = test_data['No_of_Other_ATMs_in_1_KM_radius']*test_data['Estimated_Number_of_Houses_in_1_KM_Radius']*test_data['ATM_Zone']
    test_data['ATM_Place_Attached_link_rating']=test_data["ATM_Attached_to"]*test_data['rating']*test_data['No_of_Other_ATMs_in_1_KM_radius']
    test_data['ATM_look_link_rating']=test_data["ATM_looks"]*test_data['rating']
    test_data['ATM_Type_link_rating']=test_data["ATM_TYPE"]*test_data["ATM_Placement"]*test_data['rating']
    test_data['ATM_More_choice_link_look_type']=test_data['No_of_Other_ATMs_in_1_KM_radius']+(test_data["ATM_TYPE"]*test_data['ATM_look_link_rating'])
    test_data['Wait_Time_connect_ATM_Zone']=test_data['Average_Wait_Time']*test_data['No_of_Other_ATMs_in_1_KM_radius']*test_data['ATM_Zone']*test_data['rating']

    correlations = train_data.corr()['revenue']
    selected_features = correlations[correlations > 0.2].index.tolist()

    # Remove 'revenue' from the list of features
    selected_features.remove('revenue')

    # Define the features and target
    X_train = train_data[selected_features]
    y_train = train_data['revenue']
    
    X_test_final = test_data[selected_features]
    y_test_final  = test_data['revenue']

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    regression_model= RandomForestRegressor(n_estimators=100, random_state=0)
    regression_model.fit(X_train, y_train)
    
    y_test_pred = regression_model.predict(X_test_final).astype(int)
    test_data['predicted_revenue'] = y_test_pred

    # pearson, pvalue = pearsonr(test_data['revenue'], test_data['predicted_revenue'])

    output_predicted_revenue = pd.DataFrame(y_test_pred, columns=['predicted_revenue'])
    output_predicted_revenue.to_csv('PART1.output.csv', index=False)


def ATM_Classification(train_data,test_data):
    
    LEC = LabelEncoder()
    
    train_data['ATM_Zone'] = LEC.fit_transform(train_data['ATM_Zone'])
    train_data['ATM_Placement'] = LEC.fit_transform(train_data['ATM_Placement'])
    train_data['ATM_TYPE'] = LEC.fit_transform(train_data['ATM_TYPE'])
    train_data['ATM_Location_TYPE'] = LEC.fit_transform(train_data['ATM_Location_TYPE'])
    train_data['ATM_Attached_to'] = LEC.fit_transform(train_data['ATM_Attached_to'])
    train_data['Day_Type'] = LEC.fit_transform(train_data['Day_Type'])
    train_data["ATM_looks"]=LEC.fit_transform(train_data["ATM_looks"])
    
    test_data['ATM_Zone'] = LEC.fit_transform(test_data['ATM_Zone'])
    test_data['ATM_Placement'] = LEC.fit_transform(test_data['ATM_Placement'])
    test_data['ATM_TYPE'] = LEC.fit_transform(test_data['ATM_TYPE'])
    test_data['ATM_Location_TYPE'] = LEC.fit_transform(test_data['ATM_Location_TYPE'])
    test_data['ATM_Attached_to'] = LEC.fit_transform(test_data['ATM_Attached_to'])
    test_data['Day_Type'] = LEC.fit_transform(test_data['Day_Type'])
    test_data["ATM_looks"]=LEC.fit_transform(test_data["ATM_looks"])
    
    train_data = train_data.dropna()
    test_data = test_data.dropna()
    
    train_data['Revenue_per_Shop'] = train_data['revenue'] / train_data['Number_of_Shops_Around_ATM']
    train_data['Revenue_for_location_house'] = train_data['revenue']*train_data['ATM_Location_TYPE']*train_data['Estimated_Number_of_Houses_in_1_KM_Radius']
    train_data['Revenue_for_placement'] = train_data['ATM_Placement']*train_data['revenue']*train_data['ATM_Zone'] 
    train_data['Revenue_for_attached'] = train_data['ATM_Attached_to']*train_data['revenue']
    train_data['Revenue_for_watingtime_daytype'] =train_data['Average_Wait_Time']*train_data['Day_Type']

    test_data['Revenue_per_Shop'] = test_data['revenue'] / test_data['Number_of_Shops_Around_ATM']
    test_data['Revenue_for_location_house'] = test_data['revenue']*test_data['ATM_Location_TYPE']*test_data['Estimated_Number_of_Houses_in_1_KM_Radius']
    test_data['Revenue_for_placement'] = test_data['ATM_Placement']*test_data['revenue']*test_data['ATM_Zone']
    test_data['Revenue_for_attached'] = test_data['ATM_Attached_to']*test_data['revenue']
    test_data['Revenue_for_watingtime_daytype'] =test_data['Average_Wait_Time']*test_data['Day_Type']

    X_train  =train_data[['revenue','Number_of_Shops_Around_ATM', 
                             'No_of_Other_ATMs_in_1_KM_radius', 'Estimated_Number_of_Houses_in_1_KM_Radius','Revenue_per_Shop','Revenue_for_location_house',
                                                        'Revenue_for_placement','Revenue_for_attached','Revenue_for_watingtime_daytype']]
    y_train =train_data['rating']

    X_test_final =test_data[['revenue','Number_of_Shops_Around_ATM', 
                             'No_of_Other_ATMs_in_1_KM_radius', 'Estimated_Number_of_Houses_in_1_KM_Radius','Revenue_per_Shop','Revenue_for_location_house',
                                                        'Revenue_for_placement','Revenue_for_attached','Revenue_for_watingtime_daytype']]
    y_test_final =test_data['rating']
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    Classification_model = RandomForestClassifier(n_estimators=100, random_state=0)
    Classification_model.fit(X_train, y_train)

    y_pred_test = Classification_model.predict(X_test_final)
    
#     confusion_matrix=confusion_matrix(y_test_final, y_pred_test)
#     accuracy = accuracy_score(y_test_final, y_pred_test)
#     precision = precision_score(y_test_final, y_pred_test, average='weighted')
#     recall = recall_score(y_test_final, y_pred_test, average='weighted')
#     f1 = f1_score(y_test_final, y_pred_test, average='weighted')

    output_predicted_rating = pd.DataFrame(y_pred_test, columns=['predicted_rating'])
    output_predicted_rating.to_csv('PART2.output.csv', index=False)

    
if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("Wrong inputs format, try again")
        sys.exit(-1)

    input_1=sys.argv[1]
    input_2=sys.argv[2]
    
    train_data = pd.read_csv(input_1, delimiter='\t')
    test_data = pd.read_csv(input_2, delimiter='\t')

    ATM_Regression(train_data, test_data)
    ATM_Classification(train_data,test_data)
