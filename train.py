import os
import mlflow
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm



# define parameters

params = {
    "model_name": "distilbert-base-uncased",
    "learning_rate": 5e-5,
    "batch_size": 16,
    "num_epochs": 1,
    "dataset_name": "ag_news",
    "task_name": "sequence_classification",
    "log_steps": 100,
    "max_seq_length": 128,
    "output_dir": "models/distilbert-ag-news",
}

# MLFlow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(params["task_name"])

# Start MLFlow run
mlflow.start_run(run_name=f"{params['model_name']}-{params['dataset_name']}-new")


# Log parameters
mlflow.log_params(params)

# Load dataset
dataset = load_dataset(params["dataset_name"])


tokenizer = DistilBertTokenizer.from_pretrained(params["model_name"])


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=params["max_seq_length"])

train_dataset = dataset['train'].shuffle().select(range(20_000)).map(tokenize, batched=True)
test_dataset = dataset['test'].shuffle().select(range(5_000)).map(tokenize, batched=True)

train_dataset.to_parquet("data/train.parquet")
test_dataset.to_parquet("data/test.parquet")

# log dataset artifacts
mlflow.log_artifact("data/train.parquet", artifact_path="data")
mlflow.log_artifact("data/test.parquet", artifact_path="data")


print(f"Sample of train dataset: {train_dataset[0]}")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

# get labels
labels = train_dataset.features["label"].names
print(f"Labels: {labels}")

model = DistilBertForSequenceClassification.from_pretrained(
    params["model_name"],
    num_labels=len(labels),
)
model.config.id2label = {i: label for i, label in enumerate(labels)}

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=params["learning_rate"], eps=1e-8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # calculate metrics
    accuracy_score = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    return accuracy_score, f1, precision, recall


# Training loop
with tqdm(total=params["num_epochs"] * len(train_loader), desc=f"Epoch[1/{params['num_epochs']}]") as pbar:
    for epoch in range(params["num_epochs"]):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % params["log_steps"] == 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")
                mlflow.log_metric("loss", loss.item(), step=step + epoch * len(train_loader))

            pbar.update(1)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss}")
        mlflow.log_metric("avg_loss", avg_loss, step=epoch)

        # Evaluate the model
        accuracy, f1, precision, recall = evaluate_model(model, test_loader)
        print(f"Epoch {epoch + 1} - Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")
        
        # Log evaluation metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }, step=epoch)

# Save the model
mlflow.pytorch.log_model(model, artifact_path="model")

# Log custom model 

os.makedirs(params["output_dir"], exist_ok=True)
model.save_pretrained(params["output_dir"])
tokenizer.save_pretrained(params["output_dir"])

mlflow.log_artifact(params["output_dir"], artifact_path="custom_model")


model_uri = f"runs:/{mlflow.active_run().info.run_id}/custom_model"
print(f"Model saved to: {model_uri}")

mlflow.register_model(
    model_uri=model_uri,
    name="agnews-transformer"
)
# End MLFlow run
mlflow.end_run()