from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score


class ModelEvaluation:

    def evaluate(
        self,
        model,
        X_test,
        y_test
    ):

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        print("\n Accuracy")
        print(accuracy_score(y_test, y_pred))

        print("\n classification report")
        print(classification_report(y_test, y_pred))

        print("\n confusion matrix")
        print(confusion_matrix(y_test, y_pred))

        print("\n roc-auc score")
        print(roc_auc_score(y_test, y_prob))