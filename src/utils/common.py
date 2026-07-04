import os
import joblib

def save_model(model, path):

    os.makedirs(                    # this create directory or folder to save 
        os.path.dirname(path),      # this for get the folder using the path
        exist_ok=True               # if the folder exists, no error come just continues-
    )                               # -so exists = true

    joblib.dump(model, path)
    print(f"\n model saved successfully: {path}")


def load_model(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f" model not found: {path}")

    model = joblib.load(path)

    return model