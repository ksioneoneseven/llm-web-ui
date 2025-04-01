"""
FastAPI Backend for LLM Fine-Tuning

Features:
- Upload a .txt dataset
- Pull a Hugging Face model (public or gated w/ token)
- Start training (background thread)
- CPU-only or GPU (auto) toggle
- Track training status
- List available models and datasets

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import shutil
import threading
import traceback
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from pytorch_lightning.loggers import TensorBoardLogger

# Hugging Face Hub for pulling models
from huggingface_hub import snapshot_download

# ------------------------------------------------------------------------------
# Global Config & Directories
# ------------------------------------------------------------------------------
UPLOAD_DIR = "../datasets"
MODEL_DIR = "../models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------------------
app = FastAPI(
    title="LLM Trainer Backend",
    description="A backend for uploading datasets, pulling HF models, and fine-tuning locally.",
    version="1.0.0",
)

# Allow cross-origin requests (e.g., from Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Global (thread-safe) training state
# ------------------------------------------------------------------------------
training_lock = threading.Lock()
training_in_progress = False
training_logs: Dict[str, Any] = {
    "status": "idle",
    "current_epoch": 0,
    "max_epochs": 0,
    "loss": None,
}

# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
class TextDataset(Dataset):
    """
    Splits a large text file into chunks of `block_size` tokens.
    """
    def __init__(self, tokenizer: AutoTokenizer, file_path: str, block_size: int = 512):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokenized = tokenizer(text, return_tensors="pt").input_ids[0]
        self.examples = [
            tokenized[i : i + block_size]
            for i in range(0, len(tokenized) - block_size, block_size)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.examples[idx]

def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch)

# ------------------------------------------------------------------------------
# PyTorch Lightning Module
# ------------------------------------------------------------------------------
class LLMTrainer(pl.LightningModule):
    """
    A simple LightningModule that fine-tunes a causal language model
    on next-token prediction (causal LM).
    """
    def __init__(self, model_name: str, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.train()  # 👈 REQUIRED: Enable training mode!
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lr = lr

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids, labels=input_ids)
        return outputs.loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# ------------------------------------------------------------------------------
# Background Thread: Training
# ------------------------------------------------------------------------------
def train_model(
    model_name: str,
    dataset_name: str,
    epochs: int,
    lr: float,
    block_size: int,
    use_cpu: bool
):
    """
    Fine-tune a Hugging Face model on a local text dataset using PyTorch Lightning.
    If `use_cpu` is True, force CPU training. Otherwise, use `auto` (GPU if available).
    """
    global training_in_progress, training_logs

    try:
        # Indicate that training has started
        with training_lock:
            training_in_progress = True
            training_logs["status"] = "running"
            training_logs["current_epoch"] = 0
            training_logs["max_epochs"] = epochs
            training_logs["loss"] = None

        # Prepare dataset
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset_path = os.path.join(UPLOAD_DIR, dataset_name)
        ds = TextDataset(tokenizer, dataset_path, block_size=block_size)
        dataloader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

        # Initialize the PyTorch Lightning module
        model = LLMTrainer(model_name, lr)

        # Optional: TensorBoard logger
        tb_logger = TensorBoardLogger(save_dir="lightning_logs", name="llm_training")

        # Decide CPU or GPU
        accelerator_type = "cpu" if use_cpu else "auto"

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=accelerator_type,
            logger=tb_logger,
        )

        # Callback to track epoch progress
        def epoch_end_callback(trainer_obj, pl_module):
            with training_lock:
                training_logs["current_epoch"] = trainer_obj.current_epoch

        trainer.callbacks.append(
            pl.callbacks.LambdaCallback(on_train_epoch_end=epoch_end_callback)
        )

        # Train
        trainer.fit(model, dataloader)

        # Save final model
        save_name = f"{model_name.replace('/', '-')}-{dataset_name.replace('.txt','')}"
        save_path = os.path.join(MODEL_DIR, save_name)
        model.model.save_pretrained(save_path)
        model.tokenizer.save_pretrained(save_path)

        with training_lock:
            training_logs["status"] = "completed"

    except Exception as e:
        with training_lock:
            training_logs["status"] = f"error: {e}"
    finally:
        with training_lock:
            training_in_progress = False

# ------------------------------------------------------------------------------
# FastAPI Endpoints
# ------------------------------------------------------------------------------

@app.post("/upload_dataset/")
async def upload_dataset(file: UploadFile) -> dict:
    """
    Upload a .txt file into the datasets/ directory.
    """
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": filename, "status": "uploaded"}

@app.get("/list_datasets/")
def list_datasets() -> dict:
    """
    Return a list of files in the datasets/ directory.
    """
    return {"datasets": os.listdir(UPLOAD_DIR)}

@app.get("/list_models/")
def list_models() -> dict:
    """
    Return a list of directories in the models/ directory (each presumably a model).
    """
    folders = []
    for item in os.listdir(MODEL_DIR):
        path = os.path.join(MODEL_DIR, item)
        if os.path.isdir(path):
            folders.append(item)
    return {"models": folders}

@app.post("/pull_hf_model/")
def pull_hf_model(config: dict) -> dict:
    """
    Pull a model from the Hugging Face Hub into models/.
    If the model is gated or private, you must have accepted its license
    and set HUGGINGFACE_HUB_TOKEN in your environment.

    Body example:
    {
      "model_id": "gpt2",
      "revision": "main"
    }
    """
    model_id = config.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="No model_id provided")

    revision = config.get("revision", None)
    local_dir = os.path.join(MODEL_DIR, model_id.replace("/", "-"))

    try:
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=local_dir,
            token=os.environ.get("HUGGINGFACE_HUB_TOKEN")  # <--- Use token if set
        )
        return {
            "status": "success",
            "message": f"Pulled model '{model_id}' into {local_dir}"
        }
    except Exception as e:
        print("Error pulling model:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error pulling model: {e}")

@app.post("/start_training/")
def start_training(config: dict) -> dict:
    """
    Start a new training job in a background thread.
    Body example:
    {
      "model_name": "gpt2",
      "dataset_name": "mydata.txt",
      "epochs": 3,
      "learning_rate": 0.00002,
      "block_size": 512,
      "use_cpu": false
    }
    """
    global training_in_progress

    if training_in_progress:
        return {"status": "error", "message": "Training already in progress"}

    model_name = config.get("model_name")
    dataset_name = config.get("dataset_name")
    epochs = config.get("epochs", 1)
    lr = config.get("learning_rate", 2e-5)
    block_size = config.get("block_size", 512)
    
    # New parameter for toggling CPU vs. GPU
    use_cpu = config.get("use_cpu", False)

    if not (model_name and dataset_name):
        raise HTTPException(status_code=400, detail="model_name and dataset_name are required")

    # Launch a background thread for training
    thread = threading.Thread(
        target=train_model,
        args=(model_name, dataset_name, epochs, lr, block_size, use_cpu),
        daemon=True
    )
    thread.start()

    return {"status": "training started", "model_name": model_name}

@app.get("/training_status/")
def training_status() -> dict:
    """
    Returns the current training status:
      - status: 'idle' | 'running' | 'completed' | 'error: <msg>'
      - current_epoch
      - max_epochs
      - loss (if available)
    """
    return training_logs

# If you want to run directly:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

