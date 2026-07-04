from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class DataPreprocessing:

    def preprocess(self, df):

        print("\nShape of the df:", df.shape)

        # Remove duplicate values from the dataframe, so drop.duplicates used
        df = df.drop_duplicates()

        print("After Removing duplicates:",df.shape)

        X = df.drop("Class",axis=1)             # why axis is 1, because 1 refers the column
        y = df["Class"]

        X_train,X_test,y_train,y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        joblib.dump(            #dump used to save the model, load used to read
            scaler,
            "models/scaler.pkl"
        )

        print("Training data Shape:",X_train.shape)
        print("Testing data Shape:",X_test.shape)

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )