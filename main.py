from data_fetcher import get_stock_data
from preprocess import prepare_data
from model import build_model
from predict import train_and_predict

def main():
    df = get_stock_data("AAPL")
    X, y, scaler = prepare_data(df)
    model = build_model((X.shape[1], 1))
    train_and_predict(model, X, y, scaler)

if __name__ == "__main__":
    main()
