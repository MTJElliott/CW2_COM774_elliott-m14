from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

from azure.ai.ml.entities import CommandComponent
from pathlib import Path

# Load workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="YOUR_SUB_ID",
    resource_group_name="YOUR_RESOURCE_GROUP",
    workspace_name="YOUR_WORKSPACE"
)

# Load all components
dt_comp = CommandComponent.from_yaml("train_decision_tree.yml")
lr_comp = CommandComponent.from_yaml("train_logistic_regression.yml")
rf_comp = CommandComponent.from_yaml("train_random_forest.yml")

gather_metrics_comp = CommandComponent.from_yaml("gather_metrics.yml")
register_best_comp = CommandComponent.from_yaml("register_best_models.yml")


@pipeline(default_compute="cpu-cluster")
def full_training_pipeline(training_data, testing_data):

    # ====================
    # Train Individual Models
    # ====================

    dt_run = dt_comp(
        training_data=training_data,
        testing_data=testing_data
    )

    lr_run = lr_comp(
        training_data=training_data,
        testing_data=testing_data
    )

    rf_run = rf_comp(
        training_data=training_data,
        testing_data=testing_data
    )

    # ====================
    # Gather Metrics into One File
    # ====================

    gather = gather_metrics_comp(
        dt_dir=dt_run.trained_model,
        lr_dir=lr_run.trained_model,
        rf_dir=rf_run.trained_model
    )

    # ====================
    # Register Best Models
    # ====================

    register = register_best_comp(
        metrics_file=gather.metrics_summary.path + "/all_metrics.json",
        dt_dir=dt_run.trained_model,
        lr_dir=lr_run.trained_model,
        rf_dir=rf_run.trained_model
    )

    return {
        "final_metrics": gather.metrics_summary,
        "registration_output": register.evaluation_results
    }


# Build pipeline job
pipeline_job = full_training_pipeline(
    training_data=Input(type="uri_file", path="azureml:training_data:1"),
    testing_data=Input(type="uri_file", path="azureml:testing_data:1"),
)

# Submit
ml_client.jobs.create_or_update(pipeline_job)
print("Pipeline submitted!")