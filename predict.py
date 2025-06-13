import os
import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

mlflow.set_tracking_uri("http://localhost:5000")

client = mlflow.tracking.MlflowClient() # for using advanced features like search, etc.


# Retrive the model from mlflow
model_name = "agnews_pt_classifier_new"  # Specify the model name you want to load
model_version = "1"  # Specify the version you want to load
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pytorch.load_model(model_uri)

# Sample text to predict

texts = [
    "The local football team won the championship last night.",
    "The stock market crashed today, causing widespread panic among investors.",
]

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def predict(texts, model, tokenizer):
    """
    Predict the class of the input texts using the specified model and tokenizer.
    
    Args:
        texts (list): List of input texts to classify.
        model: Pre-trained model for sequence classification.
        tokenizer: Tokenizer corresponding to the pre-trained model.
    
    Returns:
        list: Predicted classes for each input text.
    """
    # Tokenize the input texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
    predictions = predictions.tolist()
    return [model.config.id2label[pred] for pred in predictions]

# Example usage
predicted_classes = predict(texts, model, tokenizer)
print("Predicted classes: ", predicted_classes)

# Retrive custom model
custom_model_name = "agnews-transformer"
custom_model_version = "3"
model_version_details = client.get_model_version(custom_model_name, custom_model_version)
run_id = model_version_details.run_id
artifact_uri = model_version_details.source

model_path = "models/agnews-transformer" # download to local path
os.makedirs(model_path, exist_ok=True)
client.download_artifacts(run_id, artifact_uri, model_path)

custom_model = AutoModelForSequenceClassification.from_pretrained("models/agnews-transformer/custom_model/distilbert-ag-news")
tokenizer = AutoTokenizer.from_pretrained("models/agnews-transformer/custom_model/distilbert-ag-news")
custom_predicted_classes = predict(texts, custom_model, tokenizer)
print("Predicted classes from the custom model: ", custom_predicted_classes)



# Model versioning

mlflow.set_experiment("sequence_classification_new")

with mlflow.start_run(run_name="iteration2"):
    mlflow.pytorch.log_model(model, "model")

with mlflow.start_run(run_name="iteration3"):
    mlflow.pytorch.log_model(model, "model")


# Version management
model_name = "agnews_pt_classifier"
model_versions = client.search_model_versions(f"name='{model_name}'")

for model_version in model_versions:
    print(f"Version: {model_version.version} \nStage: {model_version.current_stage} \nStatus: {model_version.status}")


# Transition model stage
model_name = "agnews_pt_classifier"
model_version = "1"  # Specify the version you want to transition
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="Production"
)