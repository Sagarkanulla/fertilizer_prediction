import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = '/content/fertilizersml.csv'  # Update with actual path if necessary
df = pd.read_csv(file_path)

# Remove any trailing spaces from column names
df.columns = df.columns.str.strip()

# Correct the 'Temparature' column name if it exists
if 'Temparature' in df.columns:
    df.rename(columns={'Temparature': 'Temperature'}, inplace=True)

# Separate features and target variable
X = df.drop('Fertilizer Name', axis=1)
y = df['Fertilizer Name']

# Identify categorical and numerical columns
categorical_cols = ['Soil Type', 'Crop Type']
numerical_cols = X.columns.difference(categorical_cols)

# Preprocessing for numerical data (scaling)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data (one-hot encoding)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create the full pipeline with a classifier
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model using all possible classes
all_classes = y.unique()
report = classification_report(y_test, y_pred, labels=all_classes, target_names=all_classes)
print(report)

# Fertilizer uses dictionary
# Fertilizer uses dictionary (with exact matches for predicted outputs)
fertilizer_uses = {
    'Urea': 'Promotes fast growth and lush greenery due to high nitrogen content.',
    'DAP': 'Encourages strong root development early in plant growth.',
    'NPK(NPK(14-35-14))': 'Rich in phosphorus, ideal for crops in early stages to promote root development.',
    'NPK(NPK(28-28))': 'Balanced nitrogen and phosphorus, used at different growth stages.',
    'NPK(NPK(17-17-17))': 'All-purpose balanced fertilizer, suitable for most crops.',
    'NPK(NPK(20-20))': 'Balanced nitrogen-phosphorus fertilizer for general growth and root development.',
    'NPK(20-20)': 'Balanced nitrogen-phosphorus fertilizer for general growth and root development.',
    'NPK(17-17-17)': 'Balanced fertilizer, ideal for steady growth.',
    'NPK(28-28)': 'Good for crops in mid-stage growth with balanced nutrient requirements.',
    'NPK(14-35-14)': 'Rich in phosphorus for root strengthening.',
    'NPK(10-26-26)': 'Rich in phosphorus and potassium, ideal for flowering and fruiting stages.',
    'NPK(10=26-26)': 'Used for fruiting and flowering crops that require more phosphorus and potassium.'
}


# Function to predict fertilizer and provide its use
def predict_fertilizer(model):
    # Taking user input for the features
    temperature = float(input("Enter Temperature (e.g., 35, 36, 37, 33 in degree centigrade): "))
    humidity = float(input("Enter Humidity (e.g., 60, 70, 64 in %): "))
    moisture = float(input("Enter Moisture (e.g., 20, 30, 40 in %): "))
    soil_type = input("Enter Soil Type (e.g., Sandy, Loamy, Black, Red, Clayey): ")
    crop_type = input("Enter Crop Type (e.g., Maize, Sugarcane, Cotton, Tobacco, Paddy): ")
    nitrogen = float(input("Enter Nitrogen level (e.g., 0 to 50): "))

    # Get potassium level from the user
    potassium_input = input("Enter Potassium level (e.g., 0 to 20): ")
    potassium = float(potassium_input) if potassium_input else 0.0

    phosphorous = float(input("Enter Phosphorous level (e.g., 0 to 50): "))

    # Creating a dataframe for the input
    input_data = pd.DataFrame([[temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous]],
                              columns=['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'])

    # Making a prediction
    prediction = model.predict(input_data)[0]
    fertilizer_use = fertilizer_uses.get(prediction, "Fertilizer use information not available.")

    return f"Predicted Fertilizer: {prediction}\nUse: {fertilizer_use}"

# Call the function to take user input and make a prediction
print(predict_fertilizer(model))

# Print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
