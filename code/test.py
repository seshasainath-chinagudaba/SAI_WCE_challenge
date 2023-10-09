import os
import torch
import torchvision.transforms as transforms
from PIL import Image

from sklearn.metrics import precision_score, recall_score, f1_score

def majority_voting(probabilities, threshold):
    binary_predictions = [(prob > threshold).type(torch.int) for prob in probabilities]
    stacked_predictions = torch.stack(binary_predictions, dim=0).squeeze(dim=2)
    stacked_probabilities = torch.stack(probabilities, dim=0).squeeze(dim=2)
    mask = stacked_predictions.squeeze(dim=1).to(dtype=torch.bool)
    if mask.any():
        max_confidence_bleeding = torch.max(stacked_probabilities[mask], dim=0)[0]
    else:
        max_confidence_bleeding = None
    mask = (1 - stacked_predictions.squeeze(dim=1)).to(dtype=torch.bool)
    if mask.any():
        max_confidence_non_bleeding = torch.max(stacked_probabilities[mask], dim=0)[0]
    else:
        max_confidence_non_bleeding = None
    majority_prediction = (torch.sum(stacked_predictions, dim=0) >= (len(binary_predictions) // 2 + 1)).item()
    majority_confidence = max_confidence_bleeding.item() if majority_prediction else max_confidence_non_bleeding.item()

    return majority_prediction, majority_confidence

def test(model, test_dir):
    # Set the model to evaluation mode
    model.eval()

    # Define the transformation to preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the model's input size
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    ])

    # List all files in the test directory
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    ground_truth = [1]*len(image_files)
    threshold = 0.5

    # Loop through each image file
    majority_predictions = []
    majority_conf_probabilities = []
    max_predictions = []
    max_conf_probabilities = []
    mean_predictions = []
    mean_conf_probabilities = []

    for image_file in image_files:
        # Load and preprocess the image
        image_path = os.path.join(test_dir, image_file)
        image = Image.open(image_path)
        image_tensor = preprocess(image).unsqueeze(0)  # Add a batch dimension
        image_tensor = image_tensor.to(device)

        # Perform a forward pass through the model
        with torch.no_grad():
            outputs = model([image_tensor]*1)

            # Convert logits to probabilities using the sigmoid function
            probabilities = [torch.sigmoid(output) for output in outputs]
            #bleeding is 0
            probabilities = [1.-prob for prob in probabilities]

            mp, mc = majority_voting(probabilities, threshold)
            majority_predictions.append(mp)
            majority_conf_probabilities.append(mc)
            mp, mc = max_voting(probabilities, threshold)
            max_predictions.append(mp)
            max_conf_probabilities.append(mc)
            mp, mc = mean_voting(probabilities, threshold)
            mean_predictions.append(mp)
            mean_conf_probabilities.append(mc)
    
    """ majority_conf_probabilities = [f'{cp:.2f}' for cp in majority_conf_probabilities]
    majority_predictions_named = [label_to_name[int(p)] for p in majority_predictions]
    print("Predictions:", majority_predictions_named)
    print("confidence probs:", majority_conf_probabilities)"""

    precision = precision_score(ground_truth, majority_predictions)
    recall = recall_score(ground_truth, majority_predictions)
    f1 = f1_score(ground_truth, majority_predictions)

    print("Classification Metrics based on majority voting:")
    print(f"Precision:\t{precision:.2f}")
    print(f"Recall:\t{recall:.2f}")
    print(f"F1-score:\t{f1:.2f}")
    
    df=pd.DataFrame({'Image IDs':image_files,'predictedLabel':majority_predictions})
    df['predictedLabel'] = df['predictedLabel'].replace({True: 'bleeding', False: 'non-bleeding'})
    df.to_excel('test_data_1.xlsx',index=False)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = '/kaggle/input/modelss/WCE_class_best2198.pth'
# Load your pretrained model
model = EnsembleModel(num_models=1)

model.load_state_dict(torch.load(model_path))
model.to(device)
# Specify the directory containing test images

test_directory = '/kaggle/input/testdata/Auto-WCEBleedGen Challenge Test Dataset/Test Dataset 1'
# Call the test function
test(model, test_directory)
