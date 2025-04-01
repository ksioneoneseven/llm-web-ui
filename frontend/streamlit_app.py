import streamlit as st
import requests

# Adjust this if your FastAPI backend is on a different host/port
BACKEND_URL = "http://localhost:8000"

st.title("Local LLM Trainer Frontend")

###############################################################################
# 1. Pull Hugging Face Model
###############################################################################
st.header("1) Pull a Hugging Face Model")

hf_model_id = st.text_input("Enter a Hugging Face model ID (e.g. 'gpt2')", "")
revision = st.text_input("Optionally specify a revision (branch/tag/commit)", "main")

if st.button("Pull Model"):
    if not hf_model_id:
        st.error("Please enter a model ID first!")
    else:
        payload = {"model_id": hf_model_id, "revision": revision}
        try:
            resp = requests.post(f"{BACKEND_URL}/pull_hf_model/", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "success":
                    st.success(data.get("message"))
                else:
                    st.error(data.get("message"))
            else:
                st.error(f"Request failed with status code {resp.status_code}")
        except Exception as e:
            st.error(f"Error pulling model: {e}")

###############################################################################
# 2. Upload Dataset
###############################################################################
st.header("2) Upload a Dataset")

uploaded_file = st.file_uploader("Select a .txt file to upload", type=["txt"])

if uploaded_file is not None:
    if st.button("Upload Dataset"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
        try:
            resp = requests.post(f"{BACKEND_URL}/upload_dataset/", files=files)
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"Uploaded dataset: {data['filename']}")
            else:
                st.error(f"Upload failed with status code {resp.status_code}")
        except Exception as e:
            st.error(f"Error uploading file: {e}")

###############################################################################
# 3. List Datasets
###############################################################################
st.header("3) List Available Datasets")

if st.button("List Datasets"):
    try:
        resp = requests.get(f"{BACKEND_URL}/list_datasets/")
        if resp.status_code == 200:
            datasets = resp.json().get("datasets", [])
            if datasets:
                st.write("Datasets in /datasets:")
                for d in datasets:
                    st.write(f"- {d}")
            else:
                st.info("No datasets found.")
        else:
            st.error(f"Could not fetch datasets. Status code: {resp.status_code}")
    except Exception as e:
        st.error(f"Error listing datasets: {e}")

###############################################################################
# 4. List Models
###############################################################################
st.header("4) List Available Models")

if st.button("List Models"):
    try:
        resp = requests.get(f"{BACKEND_URL}/list_models/")
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if models:
                st.write("Models in /models:")
                for m in models:
                    st.write(f"- {m}")
            else:
                st.info("No models found.")
        else:
            st.error(f"Could not fetch models. Status code: {resp.status_code}")
    except Exception as e:
        st.error(f"Error listing models: {e}")

###############################################################################
# 5. Configure & Start Training
###############################################################################
st.header("5) Configure & Start Training")

model_name = st.text_input("Model Name/Path", "gpt2")
dataset_name = st.text_input("Dataset Filename (in /datasets)", "example.txt")
epochs = st.slider("Number of Epochs", 1, 10, 2)
learning_rate = st.number_input("Learning Rate", value=2e-5, format="%.1e")
block_size = st.slider("Block Size (tokens per sample)", 128, 2048, 512, step=128)

# New CPU-only checkbox
use_cpu = st.checkbox("Use CPU only (warning: slower)", value=False)

if st.button("Start Training"):
    if not dataset_name:
        st.error("You must specify a dataset filename.")
    else:
        payload = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "block_size": block_size,
            "use_cpu": use_cpu,  # Include the CPU-only flag
        }
        try:
            resp = requests.post(f"{BACKEND_URL}/start_training/", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "training started":
                    st.info("Training has started. Check 'Training Status' for updates.")
                else:
                    st.error(data.get("message", "Error starting training."))
            else:
                st.error(f"Failed to start training. Status code: {resp.status_code}")
        except Exception as e:
            st.error(f"Error starting training: {e}")

###############################################################################
# 6. Training Status
###############################################################################
st.header("6) Training Status")

if st.button("Refresh Training Status"):
    try:
        stat_resp = requests.get(f"{BACKEND_URL}/training_status/")
        if stat_resp.status_code == 200:
            status_data = stat_resp.json()
            st.write(status_data)
            st.write(f"Status: {status_data['status']}")
            st.write(f"Epoch: {status_data['current_epoch']} / {status_data['max_epochs']}")
            if status_data["loss"] is not None:
                st.write(f"Loss: {status_data['loss']}")
        else:
            st.error(f"Could not fetch training status. Status code: {stat_resp.status_code}")
    except Exception as e:
        st.error(f"Error fetching training status: {e}")
