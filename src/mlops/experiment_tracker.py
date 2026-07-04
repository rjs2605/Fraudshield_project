import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_score,recall_score, f1_score, roc_auc_score

class ExperimentTracker:

    def log_experiment(
        self,
        model,
        X_test,
        y_test
    ):

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        with mlflow.start_run():
            mlflow.log_param(
                "model_type",
                type(model).__name__
            )

            mlflow.log_metric("precision", precision_score(y_test, y_pred))
            mlflow.log_metric("recall", recall_score(y_test, y_pred))
            mlflow.log_metric("f1_score",f1_score(y_test, y_pred))
            mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))
            mlflow.sklearn.log_model(model, "model")

            print("\n MLflow started")