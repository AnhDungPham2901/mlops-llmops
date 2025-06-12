import mlflow
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")


experiment_name = "my_first_experiment"
mlflow.create_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id


run = mlflow.start_run(experiment_id=experiment_id, run_name="my_first_run")

# Log parameters, metrics, and artifacts
mlflow.log_param("leanring_rate", 0.01)
mlflow.log_param("batch_size", 32)


num_epochs = 10
mlflow.log_param("num_epochs", num_epochs)

for epoch in range(num_epochs):
    # Simulate training loss
    loss = np.random.rand()
    mlflow.log_metric("loss", loss, step=epoch)
    
    # Simulate accuracy
    accuracy = np.random.rand()
    mlflow.log_metric("accuracy", accuracy, step=epoch)


# log artifact

with open("data/sample.csv", "w") as f:
    f.write("x,y\n1")
    for i in range(100):
        f.write(f"{i},{np.random.rand()}\n")


mlflow.log_artifact("data/sample.csv", artifact_path="data")

# End the run
mlflow.end_run()
