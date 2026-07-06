from sklearn.model_selection import cross_val_score, StratifiedKFold

class ModelValidation:
    def validate(
        self,
        model,
        X_train,
        y_train
    ):
        skf = StratifiedKFold(
            n_splits=5,             
            shuffle=True,
            random_state=42
        )
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=skf,
            scoring="f1",
            n_jobs=-1
        )
        print("\nCross Validation Scores")
        print(scores)
        print("\n mean F1")
        print(scores.mean())
