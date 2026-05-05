import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scanpy as sc
import pickle

try:
    from .Network import scMACLLMNet
except ImportError:
    from Network import scMACLLMNet


# ------------------------
# Dataset Wrapper
# ------------------------

class SingleCellDataset(Dataset):
    """Dataset class for single-cell data after preprocessing."""

    def __init__(self, adata):
        if hasattr(adata.X, "toarray"):
            self.data = adata.X.toarray().copy().astype(np.float32)
        else:
            self.data = adata.X.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).unsqueeze(0).float()
        return x


# ------------------------
# scMAC-LLM Trainer
# ------------------------

class scMACLLMTrainer:
    """
    Trainer for scMAC-LLM model:
    - Unsupervised training via reconstruction loss
    - Latent and embedding inference
    - Loss monitoring
    """

    def __init__(self, adata, embedding=None, batch_size=64, train_epoch=20, embed_weight=1, device="cuda"):

        self.adata = adata
        self.batch_size = batch_size
        self.train_epoch = train_epoch
        self.embed_weight = embed_weight
        self.device = device
        self.loss_history = []
        self.log_messages = []

        # Handle embedding
        self.has_embedding = (
            embedding is not None
            and isinstance(embedding, torch.Tensor)
            and embedding.dim() == 2
            and embedding.shape[1] > 0
        )
        
        if embedding is not None and not self.has_embedding:
            self.log_messages.append(f"[Warning] Invalid embedding shape: {embedding.shape}, fallback to expression-only mode.")

        self.embedding = embedding.to(device) if self.has_embedding else None
        self.embed_dim = embedding.shape[1] if self.has_embedding else 3072
        self.batch_size = self.batch_size if self.has_embedding else 256  # Larger batch size is feasible to speed up computation

        if not self.has_embedding:
            self.log_messages.append("No valid embedding provided. Running in expression-only mode (scMAC).")
        else:
            self.log_messages.append("Embedding loaded. Running in the whole mode (scMAC-LLM).")

        # Model setup
        self.AE = scMACLLMNet(k=adata.shape[1], embed_dim=self.embed_dim).to(device)

        # Data loader
        dataset = SingleCellDataset(adata)
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        """Train model using MSE loss for both expression and embedding reconstruction."""
        self.AE.train()
        optimizer = torch.optim.Adam(self.AE.parameters())
        
        loss_fn = nn.MSELoss()

        for epoch in range(self.train_epoch):
            epoch_total_loss = 0.0
            epoch_expr_loss  = 0.0
            epoch_embed_loss = 0.0

            for x in self.data_loader:
                x = x.to(self.device)
                optimizer.zero_grad()

                _, _, _, _, y, ye, _ = self.AE(x, self.embedding)

                # Expression loss
                expr_loss = loss_fn(x, y)

                if self.has_embedding and ye is not None:
                    # Embedding loss with masking
                    mask = (self.embedding.abs().sum(dim=1) != 0)
                    embedding_valid = self.embedding[mask]
                    ye_valid = ye[mask]
                    if embedding_valid.numel() > 0:
                        embed_loss = loss_fn(embedding_valid, ye_valid)
                    else:
                        embed_loss = torch.tensor(0.0, device=self.device)
                else:
                    embed_loss = torch.tensor(0.0, device=self.device)

                # Total loss
                loss = expr_loss + self.embed_weight * embed_loss

                loss.backward()
                optimizer.step()

                # Accumulate losses
                epoch_total_loss += loss.item()
                epoch_expr_loss += expr_loss.item()
                epoch_embed_loss += embed_loss.item()

            # Average loss per epoch
            avg_total_loss = epoch_total_loss / len(self.data_loader)
            avg_expr_loss = epoch_expr_loss / len(self.data_loader)
            avg_embed_loss = self.embed_weight * epoch_embed_loss / len(self.data_loader)

            self.loss_history.append(avg_total_loss)

            # Log with separated losses
            log = (f"[Epoch {epoch + 1}] Total loss: {avg_total_loss:.4f} | "
                   f"Expr: {avg_expr_loss:.4f} | Embed: {avg_embed_loss:.4f}")
            self.log_messages.append(log)

    def get_train_loss_curve(self):
        """Return loss recorded at each epoch."""
        return self.loss_history

    def get_log_messages(self):
        """Return all epoch-wise training logs."""
        return self.log_messages

    def evaluate_all_outputs(self):
        """Run full forward pass and extract all output components."""
        self.AE.eval()
        dataset = SingleCellDataset(self.adata)
        # Use single batch for full-dataset inference to ensure consistency
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        with torch.no_grad():
            for x in loader:
                x = x.to(self.device)
                mean, disp, u, ue, y, ye, attn = self.AE(x, self.embedding)

        result = {
            "cell_names": self.adata.obs_names.tolist(),
            "gene_names": self.adata.var_names.tolist(),
            "cell_embed": u.cpu().numpy(),
            "recon_expr": y.cpu().squeeze(1).numpy(),
            "conv_attn": attn["scale_attn"].cpu().numpy()
        }

        if self.has_embedding and ue is not None:
            result["gene_embed"] = ue.cpu().numpy()
            result["gene_attn"] = attn["gene_attn"].cpu().numpy()
        else:
            result["gene_embed"] = None
            result["gene_attn"] = None

        return result

    @staticmethod
    def preprocess_h5ad(adata, normalize=False, flavor="seurat", n_genes=5000):
        adata = adata.copy()
        sc.pp.filter_genes(adata, min_cells=3)
        if normalize == True:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        if flavor == "seurat_v3":
            adata.X = np.round(adata.to_df()).astype(np.float32)
        sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes=n_genes)
        return adata[:, adata.var.highly_variable]

    @staticmethod
    def load_gene_embedding(adata, embedding_file):
        try:
            with open(embedding_file, 'rb') as f:
                embed_dict = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding file: {e}")

        gene_list = adata.var_names.tolist()
        embeddings = []
        found = 0

        # Automatically detect embedding dimension
        actual_dim = len(next(iter(embed_dict.values())))

        for gene in gene_list:
            if gene in embed_dict:
                vec = embed_dict[gene]
                vec = torch.tensor(vec, dtype=torch.float32) if not isinstance(vec, torch.Tensor) else vec
                found += 1
            else:
                vec = torch.zeros(actual_dim)
            embeddings.append(vec)

        log_msg = f"Matched {found} out of {len(gene_list)} genes in embedding file."
        return torch.stack(embeddings, dim=0), log_msg
