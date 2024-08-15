import mlflow

model_name = 'Keras NN'

result = mlflow.register_model(
    model_uri="runs:/fefa50007b004f248ca0a62b7174a5ce/model",  # Replace with the actual run ID or use model_uri to point to the correct model
    name=model_name
)

# Transition the model to the "Staging" stage
mlflow.tracking.MlflowClient().transition_model_version_stage(
    name=model_name,
    version=result.version,
    stage="Staging"
)