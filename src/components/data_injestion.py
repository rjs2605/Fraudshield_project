import pandas as pd

class DataIngestion:
    def __init__(self,path="data/raw/creditcard.csv"):
        self.path = path
    def load_data(self):
        df = pd.read_csv(self.path)
        print("\nDataset Loaded")
        print("Shape:", df.shape)

        return df
