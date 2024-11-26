import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModel

# Custom Dataset to handle text data
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# Collate function to tokenize a batch of texts
def collate_fn_normally(batch, model):
    # Tokenize each batch using SentenceTransformer's tokenizer
    return model.module.tokenize(batch)

def collate_fn_with_tokenizer(batch, tokenizer):
    # Tokenize each batch using a custom tokenizer
    return tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

def get_embeddings(normalized_texts, model, batch_size=526, device="cuda", tokenizer=None): 
    # Move model to the specified device and wrap in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1 and not isinstance(model, torch.nn.DataParallel):
        data_parallel = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        print(f"Using {torch.cuda.device_count()} GPUs for inference.")
    else:
        raise ValueError("Model must not be a DataParallel instance or no GPUs are available.")
    # Create DataLoader with custom collate function for batch tokenization
    dataset = TextDataset(normalized_texts)
    if tokenizer is not None:
        collate_fn_temp = lambda batch: collate_fn_with_tokenizer(batch, tokenizer)
    else:
        collate_fn_temp = lambda batch: collate_fn_normally(batch, model)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                            collate_fn=collate_fn_temp)
    
    all_embeddings = []
    with torch.no_grad():
        for tokenized_batch in tqdm(dataloader, desc="Computing embeddings"):
            # Move tokenized inputs to the specified device
            for key in tokenized_batch:
                tokenized_batch[key] = tokenized_batch[key].to(device)

            output = data_parallel(**tokenized_batch)
            # Pooling to obtain sentence embeddings; e.g., mean pooling
            batch_embeddings = output.last_hidden_state.mean(dim=1)

            # Move embeddings to CPU and append to the list
            all_embeddings.append(batch_embeddings.cpu())
    
    # Concatenate all embeddings into a single tensor if needed
    # This step is optional; if you want to save or process them individually, you can skip it
    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

def get_embeddings_distributed_async(normalized_texts, model, batch_size=526, tokenizer=None, devices=None):
    # Ensure devices list is provided; otherwise, default to all available GPUs
    if devices is None:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    print("Devices:", devices)
    
    # Load dataset and setup DataLoader
    dataset = TextDataset(normalized_texts)
    collate_fn = (lambda batch: collate_fn_with_tokenizer(batch, tokenizer)
                  if tokenizer else lambda batch: collate_fn_normally(batch, model))
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    
    model_name = model.config._name_or_path
    print(f"Using model {model_name} on {len(devices)} devices.")
    
    # Create 2 more models for each device
    models = []
    for device in devices:
        if device == "cuda:0":
            models.append(model)
        else:
            models.append(AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device))
    print("Model devices:", [m.device for m in models])
    
    # Set up CUDA streams for asynchronous execution
    streams = [torch.cuda.Stream(device=device) for device in devices]

    all_embeddings = []

    # Process each batch
    for tokenized_batch in tqdm(dataloader, desc="Computing embeddings asynchronously"):
        # Split batch data for each device
        splits = [{key: value.chunk(len(devices))[i] for key, value in tokenized_batch.items()} for i in range(len(devices))]
        
        # Initialize list to store results for each GPU
        batch_outputs = [None] * len(devices)

        # Launch asynchronous tasks on each GPU
        for i, device in enumerate(devices):
            # Use the specific stream for each device
            with torch.cuda.stream(streams[i]):
                tokenized_split = {key: value.to(device, non_blocking=True) for key, value in splits[i].items()}
                
                # Run forward pass and gather embeddings asynchronously
                with torch.no_grad():
                    output = models[i](**tokenized_split)
                    embeddings = output.last_hidden_state.mean(dim=1)
                    batch_outputs[i]= embeddings.to("cpu", non_blocking=True)
                    
                # Delete variables to free up GPU memory
                del tokenized_split, output, embeddings

        # Wait for all tasks to complete
        torch.cuda.synchronize()
        # Concatenate outputs from each GPU
        all_embeddings.append(torch.cat(batch_outputs, dim=0))
        # delete variables to free up GPU memory
        del batch_outputs
        del splits
        torch.cuda.empty_cache()

    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0)


def get_embeddings_model_parallel(normalized_texts, model, batch_size=12, tokenizer=None):
    # Set up DataLoader with tokenization
    dataset = TextDataset(normalized_texts)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda x: tokenizer(x, return_tensors="pt", padding=True, truncation=True)
    )

    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            # Send inputs to the device of the first layer of the model
            batch = {k: v.to("cuda") for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            
            # Pooling to obtain sentence embeddings; e.g., mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
            all_embeddings.append(embeddings)

    # Concatenate all embeddings to return a single tensor
    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0)

# Usage example
# device = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically select device
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# normalized_texts = ["Example sentence 1.", "Another example sentence.", "Further examples."]
# embeddings = get_embeddings(normalized_texts, model, device=device)
