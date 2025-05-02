def train_and_predict(model, X, y, scaler):
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)
    
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    y_actual = scaler.inverse_transform(y.reshape(-1, 1))

    # Save the model
    model.save("lstm_stock_model.h5")
    print("✅ Model saved as 'lstm_stock_model.h5'")

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(y_actual, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()
