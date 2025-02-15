import os
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

class_names = [
    'bacterial_leaf_blight',
    'bacterial_leaf_streak',
    'bacterial_panicle_blight',
    'blast',
    'brown_spot',
    'dead_heart',
    'downy_mildew',
    'hispa',
    'normal',
    'tungro'
]

# Load the trained model (replace 'model.pth' with your model file)
pth = 'mobilenet_v4_43_baseline_50_epochs'
model = model = timm.create_model("hf_hub:timm/mobilenetv4_conv_small.e2400_r224_in1k", pretrained=False, num_classes=10)
model = model.to(device)
model = torch.load_state_dict(torch.load(f'/Users/ananyashukla/Desktop/Ananya_Shukla/Semester 4/ILGC/low-altitude-drone/paddy-disease-classification/models/{pth}.pth', map_location=device))
model.eval()

# Define image transformations (adjust as per your model's training pipeline)
transform = transforms.Compose([
    transforms.Resize((480, 480)),  # Resize images to match model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths
test_images_dir = '/Users/ananyashukla/Desktop/Ananya_Shukla/Semester 4/ILGC/low-altitude-drone/paddy-disease-classification/test_images'
submission_file = '/Users/ananyashukla/Desktop/Ananya_Shukla/Semester 4/ILGC/low-altitude-drone/paddy-disease-classification/sample_submission.csv'

# Read the sample submission file
df = pd.read_csv(submission_file)

# Loop through each image_id, predict its label, and update the dataframe
for index, row in df.iterrows():
    image_id = row['image_id']
    image_path = os.path.join(test_images_dir, image_id)

    if os.path.exists(image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        # Predict the label
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_label = class_names[predicted.item()]

        # Update the dataframe
        df.at[index, 'label'] = predicted_label
    else:
        print(f"Image {image_id} not found in {test_images_dir}.")

# Save the updated submission file
df.to_csv(f'/Users/ananyashukla/Desktop/Ananya_Shukla/Semester 4/ILGC/low-altitude-drone/paddy-disease-classification/{pth}_submission.csv', index=False)
print("Predictions saved to sample_submission.csv")
