from flask import Flask, request
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define the model structure (same as used in training)
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # First conv layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Second conv layer
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)  # Output layer (10 classes)
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(self.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 32 * 7 * 7)  # Flatten
        x = self.relu(self.fc1(x))  # Fully connected -> ReLU
        x = self.fc2(x)  # Output layer
        return x

# Load the trained model
model = DigitClassifier()
model.load_state_dict(torch.load("digit_model.pth", map_location=torch.device("cpu")))
model.eval()

# Initialize Flask app
app = Flask(__name__)

# Function to process and predict the digit
def predict_digit(image):
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),  # Convert to tensor
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension (1, 1, 28, 28)
    output = model(image)  # Pass through the model
    return torch.argmax(output, dim=1).item()  # Return the predicted digit

@app.route("/", methods=["GET", "POST"])
def index():
    disclaimer = "Sochimazuru's digit recognizer needs grayscale images of size 28x28 to predict accurately."
    
    if request.method == "POST":
        file = request.files["file"]
        image = Image.open(file)  # Open the uploaded image
        digit = predict_digit(image)  # Use the predict_digit function to get the prediction
        return f'''
            <p>Predicted digit: {digit}</p>
            <p>{disclaimer}</p>
            <br>
            <a href="/">Try another image</a>
        '''
    
    # Display the form and disclaimer on the main page
    return f'''
        <p>{disclaimer}</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <input type="submit" value="Predict">
        </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)
