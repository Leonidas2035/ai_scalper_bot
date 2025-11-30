from dataset import SignalDataset
from model import SignalModel


def main():

    ds = SignalDataset("./data", "BTCUSDT")
    X, y = ds.create_dataset(limit=2000)

    print("Dataset size:", len(X))

    model = SignalModel()
    model.train(X, y)

    model.save("signal_model.pkl")
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
