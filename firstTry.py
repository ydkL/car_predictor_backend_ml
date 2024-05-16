
import pandas as pd

import logging
import warnings


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Redirect sklearn warnings to null handler
warnings.filterwarnings('ignore', category=Warning)


# Define the filter to ignore the specific warning message
class IgnoreSklearnWarningFilter(logging.Filter):
    def filter(self, record):
        return "X does not have valid feature names" not in record.getMessage()


ignore_sklearn_warning_filter = IgnoreSklearnWarningFilter()
logger.addFilter(ignore_sklearn_warning_filter)

# Define the data
data1 = {
    'Grid': [1, 2, 3, 4, 5, 6],
    'Model': ["Renault Megan", "Kia Ceed", "Volkswagen ID.3", "Volkswagen Golf", "Peugeot 308", "BMW 1 Series"],
}

data2 = {
    'Comfort': [5, 2, 5, 5, 1, 4],
    'Model': ["Renault Megan", "Kia Ceed", "Volkswagen ID.3", "Volkswagen Golf", "Peugeot 308", "BMW 1 Series"],
}

data3 = {
    'Tech': [1, 4, 6, 5, 5, 6],
    'Model': ["Renault Megan", "Kia Ceed", "Volkswagen ID.3", "Volkswagen Golf", "Peugeot 308", "BMW 1 Series"],
}

data4 = {
    'Visualize': [1, 2, 2, 5, 3, 6],
    'Model': ["Renault Megan", "Kia Ceed", "Volkswagen ID.3", "Volkswagen Golf", "Peugeot 308", "BMW 1 Series"],
}

data5 = {
    'Volume': [6, 5, 4, 3, 2, 2],
    'Model': ["Renault Megan", "Kia Ceed", "Volkswagen ID.3", "Volkswagen Golf", "Peugeot 308", "BMW 1 Series"],
}

data6 = {
    'Reliability': [1, 1, 5, 6, 2, 6],
    'Model': ["Renault Megan", "Kia Ceed", "Volkswagen ID.3", "Volkswagen Golf", "Peugeot 308", "BMW 1 Series"],
}

data7 = {
    'Security': [2, 2, 5, 5, 3, 6],
    'Model': ["Renault Megan", "Kia Ceed", "Volkswagen ID.3", "Volkswagen Golf", "Peugeot 308", "BMW 1 Series"],
}

data8 = {
    'Service': [2, 2, 1, 1, 3, 4],
    'Model': ["Renault Megan", "Kia Ceed", "Volkswagen ID.3", "Volkswagen Golf", "Peugeot 308", "BMW 1 Series"],
}

data9 = {
    'Insulation': [1, 2, 5, 5, 2, 6],
    'Model': ["Renault Megan", "Kia Ceed", "Volkswagen ID.3", "Volkswagen Golf", "Peugeot 308", "BMW 1 Series"],
}
'''
# Create DataFrames
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)
df4 = pd.DataFrame(data4)
df5 = pd.DataFrame(data5)
df6 = pd.DataFrame(data6)
df7 = pd.DataFrame(data7)
df8 = pd.DataFrame(data8)
df9 = pd.DataFrame(data9)

# Merge the DataFrames on 'Model'
df_merged = (df1.merge(df2, on='Model').merge(df3, on='Model').merge(df4, on='Model').merge(df5, on='Model')
             .merge(df6, on='Model').merge(df7, on='Model').merge(df8, on='Model').merge(df9, on='Model'))

# Encode the 'Model' column to numerical labels
le = LabelEncoder()
df_merged['Model_encoded'] = le.fit_transform(df_merged['Model'])

# Train the models
models = []
for feature in ['Grid', 'Comfort', 'Tech', 'Visualize', 'Volume', 'Reliability', 'Security', 'Service', "Insulation"]:
    X = df_merged[[feature]]
    y = df_merged['Model_encoded']
    model = LogisticRegression()
    model.fit(X, y)
    models.append(model)
'''


def train_models(pairs):
    dfs = []
    for m in pairs:
        dfs.append(m)
    df = pd.DataFrame(dfs[0])
    for data in dfs[1:]:
        df = pd.merge(df, pd.DataFrame(data), on='Models')

    # Encode the 'Model' column to numerical labels
    le = LabelEncoder()
    df['Model_encoded'] = le.fit_transform(df['Models'])

    X = df.drop(['Models', 'Model_encoded'], axis=1)
    y = df['Model_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model's performance
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return clf, le, X

'''
    # Train the models
    models = []
    for feature in ['Grid', 'Comfort', 'Tech', 'Visualize', 'Volume', 'Reliability', 'Security', 'Service',
                    "Insulation"]:
        X = df_merged[[feature]]
        y = df_merged['Model_encoded']
        model = LogisticRegression()
        model.fit(X, y)
        models.append(model)
        return models, le
'''

# Define function to predict car model
def predict_car_model(iv, clf, le, X):
    new_data = {
        'Grid': [iv[0]],
        'Comfort': [iv[1]],
        'Tech': [iv[2]],
        'Visualize': [iv[3]],
        'Volume': [iv[4]],
        'Reliability': [iv[5]],
        'Security': [iv[6]],
        'Service': [iv[7]],
        'Insulation': [iv[8]]
    }
    new_df = pd.DataFrame(columns=X.columns, data=[[0] * len(X.columns)])
    for col, val in new_data.items():
        new_df[col] = val

    predicted_model_encoded = clf.predict(new_df)[0]
    predicted_model = le.inverse_transform([predicted_model_encoded])[0]
    return predicted_model


'''predictions = []
for i, m in enumerate(models):
    prediction = m.predict(np.array([iv[i]]).reshape(-1, 1))
    predictions.append(prediction)
p = le.inverse_transform(predictions[-1])
return p[0]'''

'''
# easy to use
input_values = [1, 1, 1, 1, 6, 1, 1, 1, 1]

# Predict car model
predicted_model = predict_car_model(input_values)
print("Predicted car model easy to use:", predicted_model)

# grid_visualize
input_values = [6, 3, 6, 6, 3, 3, 4, 3, 3]
# Predict car model
predicted_model = predict_car_model(input_values)
print("Predicted car model grid_visualize:", predicted_model)

# comfort_security
input_values = [3, 6, 3, 3, 3, 3, 4, 3, 5]

# Predict car model
predicted_model = predict_car_model(input_values)
print("Predicted car model comfort_security:", predicted_model)'''
