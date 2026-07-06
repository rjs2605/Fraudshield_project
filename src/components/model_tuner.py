from sklearn.model_selection import GridSearchCV

class ModelTuner:
    def tune_random_forest(
        self,
        model,
        X_train,
        y_train
    ):
        params = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5]      
        }
        grid = GridSearchCV(
            estimator=model,
            param_grid=params,
            cv=3,
            scoring="f1",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        print("\n best Parameters")
        print(grid.best_params_)
        print("\n best F1")
        print(grid.best_score_)

        return grid.best_estimator_
