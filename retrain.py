import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Define the path to the saved model
model_path = "models/trained_model.pkl"

# Load the previously saved model using pickle
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
else:
    raise FileNotFoundError(f"Model not found at {model_path}")


df = pd.read_csv('data/12_30_2016_TM1.csv')
df.drop(columns=['Time'], inplace=True)

scaler = StandardScaler()
np_df = scaler.fit_transform(df)
X = np_df[:, 1:]
y = np_df[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

with open("metrics.txt", 'w') as outfile:
    outfile.write("Training error: %2.3f\n" % mse)
    outfile.write("Test variance explained: %2.3f\n" % r2)


# Define the folder path where the model will be saved
output_folder = "models"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the output file path
model_output_path = os.path.join(output_folder, "trained_model.pkl")

# Save the model to the .pkl file using pickle
with open(model_output_path, 'wb') as f:
    pickle.dump(model, f)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.savefig("losses.png",dpi=120) 
plt.close()

