
# LLM Web UI

A local setup for **fine-tuning Large Language Models (LLMs)**, featuring:

- **FastAPI** backend for:
  - Uploading datasets
  - Pulling Hugging Face models
  - Starting/stopping training jobs
  - Tracking training progress

- **Streamlit** frontend for:
  - Managing model pulls from Hugging Face
  - Uploading text datasets
  - Configuring and launching training
  - Monitoring training status

## Table of Contents
1. [Project Structure](#project-structure)
2. [Architecture Overview](#architecture-overview)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
   - [Running the Backend](#running-the-backend)
   - [Running the Frontend](#running-the-frontend)
6. [Example Workflow](#example-workflow)
7. [Future Improvements](#future-improvements)

---

## Project Structure

```
llm_web_ui/
├── backend/
│   ├── main.py           # FastAPI application
│   └──requirements.txt   # Dependencies for the backend
├── frontend/
│   ├── streamlit_app.py  # Streamlit UI
│   └── requirements.txt  # Dependencies for the frontend
├── datasets/             # Uploaded text datasets
├── models/               # Downloaded / fine-tuned models
└── README.md             # This file
```

---

## Architecture Overview

1. **FastAPI Backend** (`backend/main.py`):
   - Exposes REST endpoints for:
     - `/upload_dataset/` to receive a `.txt` file.
     - `/pull_hf_model/` to download a public model from Hugging Face Hub.
     - `/start_training/` to launch a training job in the background.
     - `/training_status/` to get the current status of the training job.
     - `/list_datasets/` & `/list_models/` to list the contents of `datasets/` and `models/`.
   - Uses **PyTorch Lightning** for training. 
   - Stores logs, checkpoints, and trained model weights in `models/`.

2. **Streamlit Frontend** (`frontend/streamlit_app.py`):
   - Provides an interactive UI for:
     - Pulling a Hugging Face model by ID.
     - Uploading a dataset.
     - Listing datasets and models.
     - Configuring training parameters (epochs, learning rate, block size, etc.).
     - Starting and monitoring the training process.
   - Communicates with the backend via **HTTP requests**.

3. **Datasets** (`datasets/`) folder:
   - Where text files are uploaded and stored.

4. **Models** (`models/`) folder:
   - Where pulled (downloaded) Hugging Face models are stored.
   - Where newly fine-tuned models are saved.

---

## Requirements

- **Python 3.8+**
- Ability to install Python packages (via `pip`).
- For GPU usage, a suitable PyTorch GPU wheel (or environment). Otherwise, CPU training is also possible (but slower).

---

## Installation

1. **Clone or download** this repository.
2. **Create a Python virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Activate on Mac/Linux:
   source venv/bin/activate
   # or on Windows:
   # venv\Scripts\activate
   ```

---

## Usage

### Running the Backend

1. **Install backend dependencies**:
   ```bash
   cd llm_web_ui/backend
   pip install -r requirements.txt
   ```
2. **Start the FastAPI server**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
3. You can test the API by visiting [http://localhost:8000/docs](http://localhost:8000/docs).

### Running the Frontend

1. **Open a new terminal** (or deactivate the backend’s virtual environment, if you prefer separate envs).
2. **Install frontend dependencies**:
   ```bash
   cd llm_web_ui/frontend
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```
4. Streamlit typically opens at [http://localhost:8501](http://localhost:8501).

---

## Example Workflow

1. **Pull a Model from Hugging Face**  
   - In your Streamlit app, under “Pull a Hugging Face Model,” enter a model ID (e.g., `gpt2`).
   - Click **Pull Model**. The model gets saved to `models/gpt2/`.

2. **Upload a Dataset**  
   - Under “Upload a Dataset,” select a `.txt` file from your local machine.
   - Once uploaded, the file appears in `datasets/`.

3. **Check Available Datasets/Models**  
   - Use “List Datasets” and “List Models” buttons to see what’s in your local folders.

4. **Configure & Start Training**  
   - Specify `model_name` (e.g., `models/gpt2` or just `gpt2`).
   - Specify `dataset_name` (the `.txt` file you uploaded).
   - Adjust `epochs`, `learning_rate`, and `block_size`.
   - Click **Start Training**.

5. **Refresh Training Status**  
   - The background training job runs in PyTorch Lightning. 
   - Click “Refresh Training Status” to see progress (epoch, total epochs, any loss info).
   - When done, the new fine-tuned model is saved under `models/<model_name>-<dataset_name>`.

---

## Future Improvements

- **Authentication & Security**  
  - For a public-facing app, add user authentication and TLS.
- **Real-Time Logging**  
  - Implement websockets or SSE for real-time loss/GPU usage charts.
- **Checkpointing & Resume**  
  - Save training checkpoints periodically so you can resume partial runs.
- **Advanced Monitoring**  
  - Integrate with W&B or TensorBoard for deeper training insights.
- **Docker**  
  - Containerize the entire stack for easy deployment (or use a Docker Compose with separate containers for backend & frontend).

---

_Thanks for using **LLM Web UI**!_  
Feel free to open issues or contribute improvements.
