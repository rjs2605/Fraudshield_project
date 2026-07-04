from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class ModelTrainer:

    def train_random_forest(
        self,
        X_train,
        y_train
    ):

        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1               # n_jobs =-1 , this uses entire cores in cpu, that's why my laptop sucks..
        )

        model.fit(X_train, y_train)

        return model

    def train_xgboost(
        self,
        X_train,
        y_train
    ):

        ratio = (y_train.value_counts()[0] / y_train.value_counts()[1])

        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=ratio,
            random_state=42,
            eval_metric="logloss"
        )

        model.fit(X_train, y_train)

        return model