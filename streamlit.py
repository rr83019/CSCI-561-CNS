import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from google.cloud import bigquery
from web3 import Web3


# ---------------------------
# PART 1: Data Loading & Preprocessing from BigQuery
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data_from_bigquery(limit=10000):
    """
    Queries the BigQuery public dataset for Ethereum transactions from the last 6 months.
    Returns a DataFrame limited to a manageable number of rows.
    """
    client = bigquery.Client()
    sql = f"""
    SELECT *
    FROM `bigquery-public-data.crypto_ethereum.transactions`
    WHERE block_timestamp >= TIMESTAMP(DATETIME_SUB(DATETIME(current_timestamp()), INTERVAL 1 MONTH))
    ORDER BY block_timestamp
    LIMIT {limit}
    """
    query_job = client.query(sql)
    df = query_job.to_dataframe()
    return df


@st.cache_data(show_spinner=False)
def load_data():
    """
    Loads Ethereum transactions from BigQuery, preprocesses the data, and creates a synthetic fraud label.
    Instead of a hard threshold on 'value', you may later use the autoencoder reconstruction error for anomaly detection.
    Here we still create a label (for evaluation) using Isolation Forest or a fixed percentile.
    For this demo, we create a synthetic label by flagging transactions with 'value' above the 90th percentile.
    """
    df = load_data_from_bigquery(limit=100000)
    if df is None or df.empty:
        st.error("No data was fetched from BigQuery.")
        return None, None, None, None, None, None

    # Define required columns.
    required_cols = ['value', 'gas', 'gas_price', 'receipt_gas_used', 'nonce', 'block_timestamp']
    if 'receipt_cumulative_gas_used' in df.columns:
        required_cols.append('receipt_cumulative_gas_used')

    missing = set(required_cols) - set(df.columns)
    if missing:
        st.error(f"Missing columns in CSV: {missing}")
        return None, None, None, None, None, None, None

    # Convert block_timestamp to datetime and extract temporal features.
    df['block_timestamp'] = pd.to_datetime(df['block_timestamp'], errors='coerce')
    df = df.dropna(subset=['block_timestamp'])
    df['hour'] = df['block_timestamp'].dt.hour
    df['day_of_week'] = df['block_timestamp'].dt.dayofweek

    # Define feature columns.
    feature_cols = ['value', 'gas', 'gas_price', 'hour', 'day_of_week', 'receipt_gas_used', 'nonce']
    if 'receipt_cumulative_gas_used' in df.columns:
        feature_cols.append('receipt_cumulative_gas_used')

    # Select features and drop rows with missing values.
    df_processed = df[feature_cols].dropna()

    # Scale features.
    scaler = StandardScaler()
    df_processed_scaled = df_processed.copy()
    df_processed_scaled[feature_cols] = scaler.fit_transform(df_processed[feature_cols])

    # Split data into training and testing sets.
    X_train, X_temp= train_test_split(
        df_processed_scaled, test_size=0.2, random_state=17
    )

    return X_train, X_temp, scaler, df_processed_scaled, df, feature_cols


# ---------------------------
# PART 2: Autoencoder Model Building & Training (PyTorch)
# ---------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Encoder with gradual compression
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),

            nn.Linear(16, 8)
        )

        # Decoder with symmetric expansion
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

@st.cache_resource(show_spinner=False)
def build_and_train_autoencoder(X_train, X_test):
    # Data preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # Convert to tensors
    train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    # Create validation split
    train_size = int(0.8 * len(train_tensor))
    val_size = len(train_tensor) - train_size
    train_dataset, val_dataset = random_split(train_tensor, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Model configuration
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    early_stop_patience = 5
    best_val_loss = float('inf')
    patience_counter = 0

    num_epochs = 1000
    train_loss_history = []
    val_loss_history = []

    model.train()
    for epoch in range(num_epochs):
        # Training phase
        epoch_train_loss = 0.0
        for batch in train_loader:
            batch_X = batch
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_X)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0)

        # Validation phase
        epoch_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch_X = batch
                outputs = model(batch_X)
                loss = criterion(outputs, batch_X)
                epoch_val_loss += loss.item() * batch_X.size(0)

        # Calculate metrics
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Calculate dynamic threshold using validation set
    model.eval()
    with torch.no_grad():
        val_reconstructions = model(torch.tensor(X_train_scaled, dtype=torch.float32))
        val_errors = torch.mean((torch.tensor(X_train_scaled) - val_reconstructions) ** 2, dim=1).numpy()

    # Use 95th percentile for tighter threshold
    threshold = np.percentile(val_errors, 99)

    # Evaluate on test set
    with torch.no_grad():
        test_reconstructions = model(test_tensor)
        test_errors = torch.mean((test_tensor - test_reconstructions) ** 2, dim=1).numpy()

    # Calculate test metrics (if labels available)
    test_accuracy = None
    test_loss = criterion(test_reconstructions, test_tensor).item()

    return model, (train_loss_history, val_loss_history), test_loss, test_accuracy, threshold


# ---------------------------
# PART 3: Live Blockchain Data Validation Using Autoencoder
# ---------------------------
def fetch_live_transactions(scaler, feature_cols):
    """
    Connects to Ethereum via Infura and fetches transactions from the latest block.
    Constructs a DataFrame with columns needed for live prediction.
    Since live transaction data may not include all fields, missing ones (e.g. receipt fields) are filled with 0.
    Extracts temporal features from block timestamp.
    """
    INFURA_PROJECT_ID = os.environ.get('INFURA_PROJECT_ID', 'YOUR_INFURA_PROJECT_ID')
    infura_url = f'https://mainnet.infura.io/v3/{INFURA_PROJECT_ID}'
    web3 = Web3(Web3.HTTPProvider(infura_url))
    if not web3.isConnected():
        st.error("Failed to connect to Ethereum network. Check your Infura Project ID.")
        return None, None

    latest_block = web3.eth.get_block('latest', full_transactions=True)
    block_ts = datetime.fromtimestamp(latest_block.timestamp)

    data_list = []
    for tx in latest_block.transactions:
        # Extract available fields.
        tx_dict = {}
        tx_dict['value'] = float(web3.fromWei(tx.value, 'ether'))
        tx_dict['gas'] = tx.gas
        tx_dict['gas_price'] = tx.gasPrice
        tx_dict['nonce'] = tx.nonce
        # For receipt_gas_used and receipt_cumulative_gas_used, use 0 if not available.
        tx_dict['receipt_gas_used'] = 0
        if 'receipt_cumulative_gas_used' in tx:
            tx_dict['receipt_cumulative_gas_used'] = tx.receipt_cumulative_gas_used
        else:
            tx_dict['receipt_cumulative_gas_used'] = 0
        # Use the block timestamp for live transactions.
        # tx_dict['block_timestamp'] = block_ts
        tx_dict['hour'] = block_ts.hour
        tx_dict['day_of_week'] = block_ts.weekday()
        data_list.append(tx_dict)

    df_live = pd.DataFrame(data_list)
    # Ensure that we include all the feature columns used in training.
    for col in feature_cols:
        if col not in df_live.columns:
            df_live[col] = 0
    # Reorder to match feature_cols.
    df_live = df_live[feature_cols]
    # Scale features using the provided scaler.
    df_live_scaled = df_live.copy()
    df_live_scaled[feature_cols] = scaler.fit_transform(df_live[feature_cols])
    X_live = df_live_scaled[feature_cols]
    return df_live, X_live


@st.cache_resource(show_spinner=False)
def predict_live_transactions(_model, _scaler, threshold, feature_cols):
    """
    Fetches live transactions, computes reconstruction error using the autoencoder,
    and flags transactions as fraud if error exceeds the threshold.
    Returns the live DataFrame with added columns for reconstruction error and fraud prediction.
    """
    df_live, X_live = fetch_live_transactions(_scaler, feature_cols)
    if df_live is None or df_live.empty:
        st.error("No live transactions available.")
        return None
    X_live_tensor = torch.tensor(X_live.values, dtype=torch.float32)
    _model.eval()
    with torch.no_grad():
        reconstructed = _model(X_live_tensor)
        errors = torch.mean((X_live_tensor - reconstructed) ** 2, dim=1).numpy()
    df_live['reconstruction_error'] = errors
    df_live['is_fraud_pred'] = (errors > threshold).astype(int)
    return df_live


# ---------------------------
# STREAMLIT APPLICATION WITH TABS
# ---------------------------
def main():
    st.title("Blockchain Security Using AI (Autoencoder & BigQuery Data)")

    # Create tabs for navigation.
    tab1, tab2, tab3 = st.tabs(["Data Preprocessing", "Train Autoencoder", "Live Blockchain Validation"])

    X_train, X_temp, scaler, df_preprocessed, df, featured_cols = load_data()

    with tab1:
        st.header("Data Loading and Preprocessing")
        if df_preprocessed is not None:
            st.write("### Original Data Preview (from BigQuery)")
            st.dataframe(df.head())
            st.write("### Columns in fetched data:", df.columns)
            st.write("### Preprocessed Data Preview")
            st.dataframe(df_preprocessed.head())

    with tab2:
        st.header("Autoencoder Training for Fraud Detection")
        if X_train is not None:
            if st.button("Train Autoencoder", key="train_autoencoder_button"):
                with st.spinner("Training the autoencoder..."):
                    model, loss_history, test_loss, test_accuracy, threshold = build_and_train_autoencoder(X_train, X_temp)
                st.success("Autoencoder trained successfully!")
                st.write("**Test Loss:**", test_loss)
                st.write("**Test Accuracy:**", test_accuracy)
                st.write("**Reconstruction Error Threshold:**", threshold)
                st.write("### Training Loss Over Epochs")
                st.line_chart(loss_history[0])
                st.write("### Validation Loss Over Epochs")
                st.line_chart(loss_history[1])

                # Generate classification metrics on the test set.
                X_test_tensor = torch.tensor(X_temp.values, dtype=torch.float32)
                with torch.no_grad():
                    test_reconstructed = model(X_test_tensor)
                    test_errors = torch.mean((X_test_tensor - test_reconstructed) ** 2, dim=1).numpy()
                y_test_pred = (test_errors > threshold).astype(int)
                # st.text("Classification Report:")
                # st.text(classification_report(y_test_pred))
                # st.text("Confusion Matrix:")
                # st.write(confusion_matrix(y_test_pred))

    with tab3:
        st.header("Live Blockchain Fraud Detection")
        if X_train is not None:
            if st.button("Run Live Prediction", key="live_prediction_button"):
                with st.spinner("Using autoencoder for live prediction..."):
                    # Train autoencoder first (or load a pre-trained model)
                    model, loss_history, test_loss, test_accuracy, threshold = build_and_train_autoencoder(X_train, X_temp)
                st.success("Autoencoder model is ready!")
                df_live = predict_live_transactions(model, scaler, threshold, featured_cols)
                if df_live is not None:
                    st.write("### Live Transactions with Fraud Prediction")
                    st.dataframe(df_live)
                else:
                    st.error("Failed to fetch or predict live transactions.")


if __name__ == "__main__":
    load_dotenv(".env")
    main()
