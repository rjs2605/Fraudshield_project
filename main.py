from src.components.data_injestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
from src.components.model_validation import ModelValidation
from src.components.model_tuner import ModelTuner
from src.components.model_evaluation import ModelEvaluation
from src.mlops.experiment_tracker import(ExperimentTracker)
from src.utils.common import save_model


def main():

    #data ingestion
    ingestion = DataIngestion()
    df = ingestion.load_data()

    #preprocessing the data
    preprocess = DataPreprocessing()
    X_train,X_test,y_train,y_test = preprocess.preprocess(df)

    #training the model
    trainer = ModelTrainer()
    model = trainer.train_random_forest(X_train, y_train)

    #Validation
    validator = ModelValidation()
    validator.validate(
        model,
        X_train,
        y_train
    )

    #hyperparameter Tuning
    tuner = ModelTuner()
    best_model = tuner.tune_random_forest(
        model,                      #this sends model to the estimator in model_tuner.py file
        X_train,
        y_train
    )

    #evaluation
    evaluator = ModelEvaluation()
    evaluator.evaluate(
        best_model,
        X_test,
        y_test
    )

    tracker = ExperimentTracker()
    tracker.log_experiment(
        best_model,
        X_test,
        y_test
    )

    # Save Model
    save_model(
        best_model,
        "models/best_random_forest.pkl"
    )


if __name__ == "__main__":
    main()