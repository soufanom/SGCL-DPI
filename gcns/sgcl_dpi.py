import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import random
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')
import os
from simgraphmaker.feature_similarity_computer import FeatureSimilarityComputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
from torch_geometric.data import Data, Batch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, classification_report
# from torch.amp import GradScaler

# Constants from previous setup
DRUG_EMBEDDING_DIM = 32  # Increased from 128
PROTEIN_EMBEDDING_DIM = 32  # Increased from 128
FUSION_DIM = 64  # Increased from 256
LEARNING_RATE = 1e-4
BATCH_SIZE = 512
NUM_EPOCHS = 200  # Increased from 100
NUM_GNN_LAYERS = 2  # Increased from 3
EARLY_STOPPING_PATIENCE = 10  # Increased from 10
SEED = 42

DRUG_COL = 'chemical'
PROTEIN_COL = 'protein'
LABEL_COL = 'label'

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

class RandomForestBaseline:
    def __init__(self, n_estimators=500):
        """Initialize Random Forest model."""
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()

    def clean_features(self, drug_features, protein_features):
        """Clean features by removing entries with NaN values."""
        print("\nCleaning features...")

        # Clean drug features
        clean_drug_features = {}
        removed_drugs = []

        for drug, (ecfp, desc) in drug_features.items():
            if isinstance(ecfp, np.ndarray) and isinstance(desc, np.ndarray):
                if not (np.isnan(ecfp).any() or np.isnan(desc).any()):
                    clean_drug_features[drug] = (ecfp, desc)
                else:
                    removed_drugs.append(drug)

        # Clean protein features
        clean_protein_features = {}
        removed_proteins = []

        for protein, (ecfp, desc) in protein_features.items():
            if isinstance(ecfp, np.ndarray) and isinstance(desc, np.ndarray):
                if not (np.isnan(ecfp).any() or np.isnan(desc).any()):
                    clean_protein_features[protein] = (ecfp, desc)
                else:
                    removed_proteins.append(protein)

        print(f"Removed {len(removed_drugs)} drugs with NaN features")
        print(f"Removed {len(removed_proteins)} proteins with NaN features")

        if removed_drugs:
            print("\nExample removed drugs:")
            for drug in removed_drugs[:5]:
                print(f"  {drug}")

        if removed_proteins:
            print("\nExample removed proteins:")
            for protein in removed_proteins[:5]:
                print(f"  {protein}")

        return clean_drug_features, clean_protein_features, removed_drugs, removed_proteins

    def extract_features(self, entity, features, entity_type):
        """Extract and concatenate features for an entity."""
        if entity_type == 'drug':
            ecfp, desc = features[entity]
        elif entity_type == 'protein':
            ecfp, desc = features[entity]
        else:
            raise ValueError(f"Unknown entity type: {entity_type}")
        return np.concatenate((ecfp, desc))

    def prepare_features(self, interactions, drug_features, protein_features):
        """Prepare features using existing feature vectors."""
        features = []
        valid_rows = []

        print("\nPreparing features from existing feature vectors...")
        total_rows = len(interactions)
        skipped_rows = 0

        # Reset index of interactions
        interactions = interactions.reset_index(drop=True)

        for idx, row in interactions.iterrows():
            try:
                drug_id = row[DRUG_COL]
                protein_id = row[PROTEIN_COL]

                # Check if features exist for both drug and protein
                if drug_id in drug_features and protein_id in protein_features:
                    try:
                        # Extract features using the provided method
                        drug_feature = self.extract_features(drug_id, drug_features, 'drug')
                        protein_feature = self.extract_features(protein_id, protein_features, 'protein')

                        # Combine features
                        combined_features = np.concatenate([drug_feature, protein_feature])
                        features.append(combined_features)
                        valid_rows.append(idx)

                    except Exception as e:
                        print(f"Error extracting features for row {idx}: {str(e)}")
                        skipped_rows += 1
                else:
                    skipped_rows += 1

            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                skipped_rows += 1

        print(f"\nFeature preparation summary:")
        print(f"Total rows: {total_rows}")
        print(f"Valid rows: {len(valid_rows)}")
        print(f"Skipped rows: {skipped_rows}")

        if len(features) == 0:
            raise ValueError("No valid features found!")

        return np.vstack(features), valid_rows

    def fit(self, train_interactions, drug_features, protein_features):
        """Train Random Forest model using existing features."""
        print("\nPreparing training data...")

        # Clean features first
        clean_drug_features, clean_protein_features, _, _ = self.clean_features(
            drug_features, protein_features
        )

        # Reset index of interactions DataFrame
        train_interactions = train_interactions.reset_index(drop=True)

        # Prepare features using cleaned feature sets
        X, valid_rows = self.prepare_features(
            train_interactions,
            clean_drug_features,
            clean_protein_features
        )

        y = train_interactions.loc[valid_rows, LABEL_COL].values

        print(f"\nTraining Random Forest...")
        print(f"Input shape: {X.shape}")
        print(f"Number of positive samples: {sum(y)}")
        print(f"Number of negative samples: {len(y) - sum(y)}")

        # Scale features
        X = self.scaler.fit_transform(X)

        # Train RF
        self.rf.fit(X, y)

        # Get predictions for training data
        train_probs = self.rf.predict_proba(X)[:, 1]

        # Initialize predictions array with zeros
        all_preds = np.zeros(len(train_interactions))

        # Fill in predictions for valid rows
        for idx, prob in zip(valid_rows, train_probs):
            all_preds[idx] = prob

        return all_preds

    def predict(self, interactions, drug_features, protein_features):
        """Get predictions using existing features."""
        print("\nPreparing test data...")

        # Clean features first
        clean_drug_features, clean_protein_features, _, _ = self.clean_features(
            drug_features, protein_features
        )

        # Reset index of interactions DataFrame
        interactions = interactions.reset_index(drop=True)

        # Prepare features using cleaned feature sets
        X, valid_rows = self.prepare_features(
            interactions,
            clean_drug_features,
            clean_protein_features
        )

        # Scale features
        X = self.scaler.transform(X)

        # Get predictions
        test_probs = self.rf.predict_proba(X)[:, 1]

        # Initialize predictions array with zeros
        all_preds = np.zeros(len(interactions))

        # Fill in predictions for valid rows
        for idx, prob in zip(valid_rows, test_probs):
            all_preds[idx] = prob

        return all_preds

class EnhancedFeatureIntegration(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Project dimensions
        hidden_dim = feature_dim // 2

        # Feature projections
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.rf_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, gnn_features, rf_pred):
        # Project features to lower dimension
        gnn_proj = self.feature_projection(gnn_features)
        rf_proj = self.rf_projection(rf_pred.unsqueeze(-1))

        # Concatenate and fuse
        combined = torch.cat([gnn_proj, rf_proj], dim=-1)
        return self.fusion_net(combined)


class EnhancedFusion(nn.Module):
    def __init__(self, drug_dim, protein_dim):
        super().__init__()

        # Dimensions
        self.hidden_dim = 256  # Reduced dimension for processing
        combined_dim = drug_dim + protein_dim

        # Projection layers
        self.drug_projection = nn.Sequential(
            nn.Linear(drug_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )

        self.protein_projection = nn.Sequential(
            nn.Linear(protein_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )

        # RF integration
        self.rf_integration = EnhancedFeatureIntegration(self.hidden_dim * 2)

        # Final prediction layers
        self.prediction_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, drug_embed, protein_embed, rf_pred):
        # Project to common dimension
        drug_hidden = self.drug_projection(drug_embed)
        protein_hidden = self.protein_projection(protein_embed)

        # Combine embeddings
        combined = torch.cat([drug_hidden, protein_hidden], dim=1)

        # Integrate RF predictions
        integrated = self.rf_integration(combined, rf_pred)

        # Final prediction
        output = self.prediction_net(integrated)

        return output.squeeze(-1), (drug_hidden, protein_hidden)

class EnhancedGNNModel(nn.Module):
    def __init__(self, drug_input_dim, protein_input_dim):
        super().__init__()

        # Model dimensions
        self.gnn_hidden_dim = 128
        self.num_layers = 3

        # Graph encoders
        self.drug_encoder = GNNEncoder(
            input_dim=drug_input_dim,
            hidden_dim=self.gnn_hidden_dim,
            num_layers=self.num_layers
        )

        self.protein_encoder = GNNEncoder(
            input_dim=protein_input_dim,
            hidden_dim=self.gnn_hidden_dim,
            num_layers=self.num_layers
        )

        # Attention layers
        self.drug_attention = nn.Sequential(
            nn.Linear(self.gnn_hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.protein_attention = nn.Sequential(
            nn.Linear(self.gnn_hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.drug_predictor = nn.Sequential(
            nn.Linear(self.gnn_hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        self.protein_predictor = nn.Sequential(
            nn.Linear(self.gnn_hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        # Fusion module
        self.fusion = EnhancedFusion(
            drug_dim=self.gnn_hidden_dim,
            protein_dim=self.gnn_hidden_dim
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with smaller bounds for stability."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, GCNConv)):
                # Use a smaller gain for initialization
                nn.init.xavier_uniform_(m.weight if isinstance(m, nn.Linear) else m.lin.weight,
                                        gain=0.1)  # Reduced from default gain=1.0
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
                elif hasattr(m, 'lin') and m.lin.bias is not None:
                    nn.init.zeros_(m.lin.bias)

            # Special handling for LayerNorm
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, drug_data, protein_data, drug_indices, protein_indices, rf_pred):
        """
        Forward pass with full graph attention.

        Args:
            drug_data: Full drug graph data
            protein_data: Full protein graph data
            drug_indices: Indices of drugs in batch (batch_size,)
            protein_indices: Indices of proteins in batch (batch_size,)
            rf_pred: Random forest predictions for batch (batch_size,)

        Returns:
            output: Final predictions
            (drug_embed, protein_embed, attention_weights): Intermediate representations
        """
        try:
            batch_size = len(drug_indices)

            # Debug protein data
            # print("\nProtein Data Statistics:")
            # print(f"Protein node features (x) shape: {protein_data.x.shape}")
            # print(
            #     f"Protein node features stats: min={protein_data.x.min():.4f}, max={protein_data.x.max():.4f}, mean={protein_data.x.mean():.4f}")
            # print(f"Number of protein edges: {protein_data.edge_index.shape[1]}")
            # print(f"Edge attributes shape: {protein_data.edge_attr.shape}")
            # print(
            #     f"Edge attributes stats: min={protein_data.edge_attr.min():.4f}, max={protein_data.edge_attr.max():.4f}, mean={protein_data.edge_attr.mean():.4f}")

            # Check for NaN values - debugging
            # print(f"NaN in protein node features: {torch.isnan(protein_data.x).any()}")
            # print(f"NaN in protein edge attributes: {torch.isnan(protein_data.edge_attr).any()}")

            # Get node embeddings for full graphs
            drug_node_embeddings = self.drug_encoder(
                drug_data.x,
                drug_data.edge_index,
                drug_data.edge_attr
            )

            if torch.isnan(drug_node_embeddings).any():
                print("NaN in drug_node_embeddings after encoder")
                drug_node_embeddings = torch.nan_to_num(drug_node_embeddings, 0.0)

            protein_node_embeddings = self.protein_encoder(
                protein_data.x,
                protein_data.edge_index,
                protein_data.edge_attr
            )

            # For debugging
            # print("\nBefore attention:")
            # print(
            #     f"Drug embeddings: min={drug_node_embeddings.min():.4f}, max={drug_node_embeddings.max():.4f}, mean={drug_node_embeddings.mean():.4f}")
            # print(
            #     f"Protein embeddings: min={protein_node_embeddings.min():.4f}, max={protein_node_embeddings.max():.4f}, mean={protein_node_embeddings.mean():.4f}")

            if torch.isnan(drug_node_embeddings).any():
                print("NaN in drug_node_embeddings after encoder")
                drug_node_embeddings = torch.nan_to_num(drug_node_embeddings, 0.0)

            # Create attention masks
            drug_mask = torch.zeros(
                batch_size,
                drug_node_embeddings.size(0),
                device=drug_node_embeddings.device
            )
            drug_mask.scatter_(1, drug_indices.unsqueeze(1), 1)

            protein_mask = torch.zeros(
                batch_size,
                protein_node_embeddings.size(0),
                device=protein_node_embeddings.device
            )
            protein_mask.scatter_(1, protein_indices.unsqueeze(1), 1)

            # For debugging
            # print("\nMask statistics:")
            # print(f"Drug mask sum: {drug_mask.sum().item()} (should equal batch_size={batch_size})")
            # print(f"Drug mask shape: {drug_mask.shape}")
            # print(f"Protein mask sum: {protein_mask.sum().item()} (should equal batch_size={batch_size})")
            # print(f"Protein mask shape: {protein_mask.shape}")

            # Compute attention scores with gradient clipping
            drug_attention = self.drug_attention(drug_node_embeddings)
            protein_attention = self.protein_attention(protein_node_embeddings)

            # For debugging
            # print("\nAttention scores before masking:")
            # print(
            #     f"Drug attention: min={drug_attention.min():.4f}, max={drug_attention.max():.4f}, mean={drug_attention.mean():.4f}")
            # print(
            #     f"Protein attention: min={protein_attention.min():.4f}, max={protein_attention.max():.4f}, mean={protein_attention.mean():.4f}")

            drug_attention = torch.clamp(drug_attention, min=1e-6, max=1 - 1e-6)
            protein_attention = torch.clamp(protein_attention, min=1e-6, max=1 - 1e-6)

            # Mask and normalize attention scores
            drug_attention = drug_attention.transpose(0, 1)  # (1, num_nodes)
            protein_attention = protein_attention.transpose(0, 1)  # (1, num_nodes)

            # Apply masked attention
            drug_masked_attention = drug_mask * drug_attention  # (batch_size, num_nodes)
            protein_masked_attention = protein_mask * protein_attention  # (batch_size, num_nodes)

            # For debugging
            # print("\nMasked attention before softmax:")
            # print(
            #     f"Drug masked attention: min={drug_masked_attention.min():.4f}, max={drug_masked_attention.max():.4f}, mean={drug_masked_attention.mean():.4f}")
            # print(
            #     f"Protein masked attention: min={protein_masked_attention.min():.4f}, max={protein_masked_attention.max():.4f}, mean={protein_masked_attention.mean():.4f}")

            # Add small epsilon to prevent division by zero
            eps = 1e-8
            drug_masked_attention = F.softmax(drug_masked_attention + eps, dim=1)
            protein_masked_attention = F.softmax(protein_masked_attention + eps, dim=1)

            # Get attended embeddings
            drug_embed = torch.matmul(drug_masked_attention, drug_node_embeddings)  # (batch_size, hidden_dim)
            protein_embed = torch.matmul(protein_masked_attention, protein_node_embeddings)  # (batch_size, hidden_dim)

            # For debugging
            # print("\nFinal embeddings:")
            # print(f"Drug embed: min={drug_embed.min():.4f}, max={drug_embed.max():.4f}, mean={drug_embed.mean():.4f}")
            # print(
            #     f"Protein embed: min={protein_embed.min():.4f}, max={protein_embed.max():.4f}, mean={protein_embed.mean():.4f}")

            # Debug attended embeddings
            if self.training and torch.isnan(drug_embed).any():
                print("Warning: NaN values in attended drug embeddings")
            if self.training and torch.isnan(protein_embed).any():
                print("Warning: NaN values in attended protein embeddings")

            # Check and fix NaN values
            drug_embed = torch.nan_to_num(drug_embed, 0.0)
            protein_embed = torch.nan_to_num(protein_embed, 0.0)

            # Get individual predictions from embeddings
            drug_pred = self.drug_predictor(drug_embed)  # Should output [batch_size, 1]
            protein_pred = self.protein_predictor(protein_embed)  # Should output [batch_size, 1]

            # Get fusion predictions
            output, attention_weights = self.fusion(
                drug_embed,
                protein_embed,
                rf_pred
            )

            return output, (drug_pred, protein_pred, attention_weights)

        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


class AUCLoss(nn.Module):
    """Differentiable AUC loss."""

    def __init__(self, gamma=0.3):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        pos_pred = pred[target == 1]
        neg_pred = pred[target == 0]

        if len(pos_pred) == 0 or len(neg_pred) == 0:
            return torch.tensor(0.0, device=pred.device)

        pos_pred = pos_pred.unsqueeze(0)
        neg_pred = neg_pred.unsqueeze(1)

        difference = pos_pred - neg_pred
        loss = torch.sigmoid(-difference / self.gamma)

        return loss.mean()


class EnhancedCombinedLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.auc_loss = AUCLoss()

    def forward(self, output, target, rf_pred, drug_pred, protein_pred, alpha=35, beta=1.0, lambda_auc=0):
        # Main BCE loss with sample weighting
        bce_loss = self.bce(output.squeeze(), target.float())

        with torch.no_grad():
            pred_diff = torch.abs(torch.sigmoid(output.squeeze()) - target.float())
            rf_confidence = torch.abs(rf_pred - 0.5)
            weights = 1.0 + torch.exp(-pred_diff * 5.0) * rf_confidence

            # Calculate current batch metrics for dynamic weighting
            current_preds = (torch.sigmoid(output.squeeze()) > 0.5).float()
            true_positives = (current_preds * target).sum()
            predicted_positives = current_preds.sum() + 1e-7  # avoid division by zero
            actual_positives = target.sum() + 1e-7  # avoid division by zero

            batch_precision = true_positives / predicted_positives
            batch_recall = true_positives / actual_positives

            # Create dynamic weights based on current performance
            precision_weight = torch.exp(-batch_precision)  # Higher weight when precision is low
            recall_weight = torch.exp(-batch_recall)  # Higher weight when recall is low

        weighted_bce = (bce_loss * weights).mean()

        # AUC optimization loss
        auc_loss = self.auc_loss(output.squeeze(), target)

        # Dual threshold knowledge distillation loss with dynamic weighting
        rf_medium_confidence = rf_pred > 0.4
        rf_high_confidence = rf_pred > 0.8

        kd_loss_medium = F.mse_loss(
            torch.sigmoid(output.squeeze())[rf_medium_confidence],
            rf_pred[rf_medium_confidence]
        ) if rf_medium_confidence.any() else torch.tensor(0.0).to(output.device)

        kd_loss_high = F.mse_loss(
            torch.sigmoid(output.squeeze())[rf_high_confidence],
            rf_pred[rf_high_confidence]
        ) if rf_high_confidence.any() else torch.tensor(0.0).to(output.device)

        # Apply dynamic weighting to KD losses
        kd_loss = (precision_weight * kd_loss_medium +
                   recall_weight * kd_loss_high) / (precision_weight + recall_weight)

        # Structure preservation loss
        structure_loss = self._compute_structure_loss(drug_pred, protein_pred, target)

        # Combine all losses with dynamic weighting
        total_loss = (weighted_bce +
                      alpha * kd_loss +
                      beta * structure_loss)

        return {
            'total_loss': total_loss,
            'weighted_bce': weighted_bce.item(),
            'auc_loss': auc_loss.item(),
            'kd_loss': kd_loss.item(),
            'structure_loss': structure_loss.item(),
            'batch_precision': batch_precision.item(),
            'batch_recall': batch_recall.item()
        }

    def precision_penalty_loss(self, student_probs, true_labels, beta=2):
        """Penalize false positives to improve precision."""
        false_positives = ((student_probs > 0.5) & (true_labels == 0)).float().sum()
        penalty = beta * false_positives
        return penalty

    def _compute_kd_loss(self, student_logits, teacher_probs, true_labels, temperature):
        student_probs = torch.sigmoid(student_logits / temperature)
        teacher_probs = teacher_probs.unsqueeze(1)

        # Standard kd_loss
        kd_loss = F.mse_loss(student_probs, teacher_probs) * (temperature ** 2)

        # Precision penalty
        precision_loss = self.precision_penalty_loss(student_probs, true_labels)

        kid_loss = kd_loss + precision_loss

        return kid_loss

    def _compute_structure_loss(self, drug_pred, protein_pred, target):
        # Add class weights
        pos_weight = (len(target) - target.sum()) / target.sum()

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Individual losses with stronger regularization
        drug_loss = criterion(drug_pred.squeeze(), target.float())
        protein_loss = criterion(protein_pred.squeeze(), target.float())

        # L2 regularization for predictor weights
        l2_reg = 0.0
        for name, param in self.named_parameters():
            if 'predictor' in name and 'weight' in name:
                l2_reg += torch.norm(param)

        return (drug_loss + protein_loss) / 2 + 0.01 * l2_reg

class CurriculumTrainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, device,
                 max_iterations=500,
                 lr=1e-4,
                 warmup_epochs=5,
                 checkpoint_dir='checkpoints',
                 drug_graph_data=None,
                 protein_graph_data=None):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.checkpoint_dir = checkpoint_dir

        # Store graph data
        self.drug_graph_data = drug_graph_data
        self.protein_graph_data = protein_graph_data

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Cosine annealing scheduler with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=warmup_epochs,
            T_mult=2
        )

        # Initialize criterion
        self.criterion = EnhancedCombinedLoss()

        # Define curriculum stages
        self.stages = [
            {
                'epochs': 10,
                'alpha': 35,  # Strong RF guidance
                'beta': 1,  # Light structure preservation
                'lambda_auc': 0
            }
        ]

        self.best_metrics = {
            'auc_roc': 0.0,
            'auprc': 0.0,
            'epoch': 0,
            'iteration': 0
        }

    def train_epoch(self, stage_params):
        """Training epoch with full graph attention."""
        self.model.train()
        epoch_losses = []
        loss_components = {
            'weighted_bce': 0.0,
            'auc_loss': 0.0,
            'kd_loss': 0.0,
            'structure_loss': 0.0
        }

        # Initialize gradient scaler
        #scaler = GradScaler()

        try:
            for batch_idx, (drug_indices, protein_indices, rf_pred, labels) in enumerate(self.train_loader):
                # Check iteration limit
                if self.current_iteration >= self.max_iterations:
                    print(f"\nReached iteration limit of {self.max_iterations}")
                    if epoch_losses:
                        avg_loss = np.mean(epoch_losses)
                        avg_components = {k: v / len(epoch_losses) for k, v in loss_components.items()}
                        return avg_loss, avg_components, True
                    return None, None, True

                # Move batch data to device
                drug_indices = drug_indices.to(self.device)
                protein_indices = protein_indices.to(self.device)
                rf_pred = rf_pred.to(self.device)
                labels = labels.to(self.device)

                # Forward pass with full graphs
                self.optimizer.zero_grad()
                # with torch.autocast(device_type='cuda', dtype=torch.float32):
                outputs, (drug_embed, protein_embed, attention_weights) = self.model(
                    self.drug_graph_data,
                    self.protein_graph_data,
                    drug_indices,
                    protein_indices,
                    rf_pred
                )

                # Compute losses
                losses = self.criterion(
                    outputs,
                    labels,
                    rf_pred,
                    drug_embed,
                    protein_embed,
                    alpha=stage_params['alpha'],
                    beta=stage_params['beta'],
                    lambda_auc=stage_params['lambda_auc']
                )

                # Backward pass
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # scaler.scale(losses['total_loss']).backward()
                # # Unscale gradients and clip
                # scaler.unscale_(self.optimizer)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                #
                # # Update weights with scaled gradients
                # scaler.step(self.optimizer)
                # scaler.update()

                # Update loss tracking
                epoch_losses.append(losses['total_loss'].item())
                for key in loss_components:
                    loss_components[key] += losses[key]

                # Print progress
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}/{len(self.train_loader)}, "
                          f"Iteration {self.current_iteration}/{self.max_iterations}, "
                          f"Loss: {losses['total_loss'].item():.4f}")

                self.current_iteration += 1

                # Clear GPU memory periodically
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()

            # Return average losses
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                avg_components = {k: v / len(self.train_loader) for k, v in loss_components.items()}
                return avg_loss, avg_components, False

            return None, None, True

        except Exception as e:
            print(f"Error in training: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, True

    def evaluate(self, loader, phase="validation"):
        """Evaluate model with full graph attention."""
        self.model.eval()
        all_preds = []
        all_labels = []

        print(f"\nStarting {phase} evaluation...")

        try:
            with torch.no_grad():
                for batch_idx, (drug_indices, protein_indices, rf_pred, labels) in enumerate(loader):
                    # Move to device
                    drug_indices = drug_indices.to(self.device)
                    protein_indices = protein_indices.to(self.device)
                    rf_pred = rf_pred.to(self.device)

                    # Get model predictions using full graphs
                    outputs, _ = self.model(
                        self.drug_graph_data,
                        self.protein_graph_data,
                        drug_indices,
                        protein_indices,
                        rf_pred
                    )

                    # Handle different output shapes
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                    # Apply sigmoid to get probabilities
                    predictions = torch.sigmoid(outputs.squeeze())

                    # Handle single item case
                    if predictions.dim() == 0:
                        predictions = predictions.unsqueeze(0)

                    # Move to CPU and convert to numpy
                    pred_np = predictions.cpu().detach().numpy()
                    label_np = labels.cpu().numpy()

                    # Validate predictions
                    if np.isnan(pred_np).any():
                        print(f"Warning: NaN predictions in batch {batch_idx}")
                        continue

                    # Debug information for first batch
                    if batch_idx == 0:
                        print(f"\nDebug info for first {phase} batch:")
                        print(f"Predictions shape: {pred_np.shape}")
                        print(f"Labels shape: {label_np.shape}")
                        print(f"Prediction range: [{pred_np.min():.4f}, {pred_np.max():.4f}]")
                        print(f"Unique labels: {np.unique(label_np)}")
                        print(f"Label distribution: {np.bincount(label_np.astype(int))}")

                    all_preds.extend(pred_np)
                    all_labels.extend(label_np)

                    # Clear GPU memory periodically
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()

                # Convert to numpy arrays
                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)

                # Final validation
                if len(all_preds) == 0 or len(all_labels) == 0:
                    print(f"Error: No valid predictions or labels in {phase} set")
                    return {metric: 0.0 for metric in
                            ['auc_roc', 'auprc', 'accuracy', 'precision', 'recall', 'f1']}

                # Print evaluation summary
                print(f"\n{phase.capitalize()} Evaluation Summary:")
                print(f"Total samples: {len(all_labels)}")
                print(f"Positive samples: {np.sum(all_labels == 1)}")
                print(f"Negative samples: {np.sum(all_labels == 0)}")
                print(f"Prediction range: [{all_preds.min():.4f}, {all_preds.max():.4f}]")

                # Compute and return metrics
                metrics = self._compute_metrics(all_preds, all_labels)

                # Print detailed metrics
                print(f"\n{phase.capitalize()} Metrics:")
                for metric, value in metrics.items():
                    print(f"{metric}: {value:.4f}")

                return metrics

        except Exception as e:
            print(f"\nError in {phase} evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            return {metric: 0.0 for metric in
                    ['auc_roc', 'auprc', 'accuracy', 'precision', 'recall', 'f1']}
        finally:
            # Final GPU memory cleanup
            torch.cuda.empty_cache()

    def _compute_metrics(self, predictions, labels):
        """Compute evaluation metrics with validation and debugging."""
        try:
            # Validate inputs
            if len(predictions) != len(labels):
                raise ValueError(f"Length mismatch: predictions ({len(predictions)}) vs labels ({len(labels)})")

            if not (labels.dtype in [np.int32, np.int64, np.float32, np.float64]):
                raise ValueError(f"Invalid label dtype: {labels.dtype}")

            # Ensure binary labels
            labels = labels.astype(np.int64)
            if not set(np.unique(labels)).issubset({0, 1}):
                raise ValueError(f"Invalid label values: {np.unique(labels)}")

            # Ensure predictions are probabilities
            if np.any(predictions < 0) or np.any(predictions > 1):
                print("Warning: Predictions outside [0,1] range, clipping values")
                predictions = np.clip(predictions, 0, 1)

            # Get binary predictions for metrics that need them
            predictions_binary = (predictions >= 0.5).astype(np.int64)

            # Compute metrics with error handling
            metrics = {}

            # Each metric in try-except block
            try:
                metrics['accuracy'] = accuracy_score(labels, predictions_binary)
            except Exception as e:
                print(f"Error computing accuracy: {e}")
                metrics['accuracy'] = 0.0

            try:
                metrics['precision'] = precision_score(labels, predictions_binary)
            except Exception as e:
                print(f"Error computing precision: {e}")
                metrics['precision'] = 0.0

            try:
                metrics['recall'] = recall_score(labels, predictions_binary)
            except Exception as e:
                print(f"Error computing recall: {e}")
                metrics['recall'] = 0.0

            try:
                metrics['f1'] = f1_score(labels, predictions_binary)
            except Exception as e:
                print(f"Error computing F1: {e}")
                metrics['f1'] = 0.0

            try:
                metrics['auc_roc'] = roc_auc_score(labels, predictions)
            except Exception as e:
                print(f"Error computing AUC-ROC: {e}")
                metrics['auc_roc'] = 0.0

            try:
                metrics['auprc'] = average_precision_score(labels, predictions)
            except Exception as e:
                print(f"Error computing AUPRC: {e}")
                metrics['auprc'] = 0.0

            # Print detailed metrics information
            print("\nDetailed Metrics Information:")
            print(f"Confusion Matrix:\n{confusion_matrix(labels, predictions_binary)}")
            print("\nClassification Report:")
            print(classification_report(labels, predictions_binary))

            return metrics

        except Exception as e:
            print(f"Error in metric computation: {str(e)}")
            import traceback
            traceback.print_exc()
            return {metric: 0.0 for metric in
                    ['auc_roc', 'auprc', 'accuracy', 'precision', 'recall', 'f1']}

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save a checkpoint of the model."""
        checkpoint = {
            'epoch': epoch,
            'iteration': self.current_iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_metrics': self.best_metrics
        }

        # Save best model checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"\nBest model saved to: {best_path}")

        return best_path if is_best else None

    def train(self, patience=10):
        """Full training loop with curriculum learning and iteration limit."""
        print("\nStarting training...")
        try:
            best_metrics_overall = {
                'auc_roc': 0.0,
                'auprc': 0.0,
                'epoch': 0,
                'stage': 0,
                'iteration': 0
            }

            for stage_idx, stage in enumerate(self.stages):
                print(f"\nStarting training stage {stage_idx + 1}")
                print(f"Parameters: {stage}")

                self.current_iteration = 0
                patience_counter = 0
                best_metrics_stage = {
                    'auc_roc': 0.0,
                    'auprc': 0.0,
                    'epoch': 0,
                    'iteration': 0
                }

                for epoch in range(stage['epochs']):
                    # Training
                    avg_loss, loss_components, hit_limit = self.train_epoch(stage)
                    if avg_loss is None:  # Training error occurred
                        print(f"\nTraining error in epoch {epoch + 1} of stage {stage_idx + 1}")
                        break

                    # Validation
                    val_metrics = self.evaluate(self.valid_loader, "validation")

                    # Update learning rate
                    self.scheduler.step()

                    # Print epoch results
                    print(f"\nStage {stage_idx + 1}, Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
                    print("Validation Metrics:",
                          ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()))

                    # Check for improvement in current stage
                    improved = (val_metrics['auc_roc'] > best_metrics_stage['auc_roc'] or
                                val_metrics['auprc'] > best_metrics_stage['auprc'])

                    if improved:
                        best_metrics_stage = {
                            'auc_roc': max(val_metrics['auc_roc'], best_metrics_stage['auc_roc']),
                            'auprc': max(val_metrics['auprc'], best_metrics_stage['auprc']),
                            'epoch': epoch,
                            'iteration': self.current_iteration
                        }
                        patience_counter = 0

                        # Update overall best metrics if better
                        if (val_metrics['auc_roc'] > best_metrics_overall['auc_roc'] or
                                val_metrics['auprc'] > best_metrics_overall['auprc']):
                            best_metrics_overall = {
                                'auc_roc': max(val_metrics['auc_roc'], best_metrics_overall['auc_roc']),
                                'auprc': max(val_metrics['auprc'], best_metrics_overall['auprc']),
                                'epoch': epoch,
                                'stage': stage_idx + 1,
                                'iteration': self.current_iteration
                            }
                            self.save_checkpoint(epoch, val_metrics, is_best=True)
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"\nEarly stopping triggered in stage {stage_idx + 1}")
                            break

                    if hit_limit:
                        print(f"\nReached iteration limit in epoch {epoch + 1} of stage {stage_idx + 1}")
                        break

                print(f"\nStage {stage_idx + 1} completed:")
                print(f"Best stage metrics - AUC ROC: {best_metrics_stage['auc_roc']:.4f}, "
                      f"AUPRC: {best_metrics_stage['auprc']:.4f}")
                print(f"Best stage epoch: {best_metrics_stage['epoch'] + 1}")
                print(f"Best stage iteration: {best_metrics_stage['iteration']}")

            # Final evaluation
            print("\nTraining completed. Performing final evaluation...")
            print("\nBest overall metrics:")
            print(f"Stage: {best_metrics_overall['stage']}")
            print(f"Epoch: {best_metrics_overall['epoch'] + 1}")
            print(f"Iteration: {best_metrics_overall['iteration']}")
            print(f"AUC ROC: {best_metrics_overall['auc_roc']:.4f}")
            print(f"AUPRC: {best_metrics_overall['auprc']:.4f}")

            best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("\nLoaded best model for final evaluation")

            final_val_metrics = self.evaluate(self.valid_loader, "validation")
            final_test_metrics = self.evaluate(self.test_loader, "test")

            return {
                'validation': final_val_metrics,
                'test': final_test_metrics,
                'best_validation': best_metrics_overall
            }

        except Exception as e:
            print(f"\nError during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'validation': best_metrics_overall,
                'test': None,
                'best_validation': best_metrics_overall
            }


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(drug_drug_file, protein_protein_file, train_interaction_file, test_interaction_file, features_file):
    drug_drug_scores = pd.read_csv(drug_drug_file)
    protein_protein_scores = pd.read_csv(protein_protein_file)
    train_interactions = pd.read_csv(train_interaction_file)
    test_interactions = pd.read_csv(test_interaction_file)
    with open(features_file, 'rb') as f:
        features = pickle.load(f)
    return drug_drug_scores, protein_protein_scores, train_interactions, test_interactions, features


def extract_features(entity, features, entity_type):
    if entity_type == 'drug':
        ecfp, desc = features['drug_features'][entity]
    elif entity_type == 'protein':
        ecfp, desc = features['protein_features'][entity]
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")
    return np.concatenate((ecfp, desc))


def create_or_load_graphs(processor, cache_file='cached_graphs.pkl'):
    """Create or load cached graphs with validation."""
    # Ensure we're on CPU for data preparation
    device = torch.device('cpu')

    def validate_graph(graph, graph_type="drug"):
        """Validate graph while ensuring CPU operations."""
        # Ensure graph is on CPU for validation
        graph = graph.to('cpu')

        print(f"\nValidating {graph_type} graph:")
        print(f"Number of nodes: {graph.num_nodes}")
        print(f"Number of edges: {graph.num_edges}")
        print(f"Node features shape: {graph.x.shape}")
        print(f"Node features stats:")
        print(f"- Min: {graph.x.min().item()}")
        print(f"- Max: {graph.x.max().item()}")
        print(f"- Mean: {graph.x.mean().item()}")
        print(f"- NaN count: {torch.isnan(graph.x).sum().item()}")

        if hasattr(graph, 'edge_attr'):
            print(f"Edge features stats:")
            print(f"- Min: {graph.edge_attr.min().item()}")
            print(f"- Max: {graph.edge_attr.max().item()}")
            print(f"- Mean: {graph.edge_attr.mean().item()}")
            print(f"- NaN count: {torch.isnan(graph.edge_attr).sum().item()}")

        # Check for isolated nodes
        row, col = graph.edge_index
        degree = torch.bincount(row, minlength=graph.num_nodes)
        print(f"Node degree distribution:")
        print(f"- Min degree: {degree.min().item()}")
        print(f"- Max degree: {degree.max().item()}")
        print(f"- Mean degree: {degree.float().mean().item()}")
        print(f"- Isolated nodes: {(degree == 0).sum().item()}")

        # Check edge index validity
        max_idx = graph.edge_index.max()
        if max_idx >= graph.num_nodes:
            print(f"WARNING: Edge index contains invalid node references!")
            print(f"Max index: {max_idx}, Number of nodes: {graph.num_nodes}")

        # Check for self loops
        self_loops = (graph.edge_index[0] == graph.edge_index[1]).sum()
        print(f"Number of self-loops: {self_loops.item()}")

        # Check for disconnected components
        edge_list = graph.edge_index.cpu().t().tolist()
        G = nx.Graph(edge_list)
        num_components = nx.number_connected_components(G)
        largest_cc = len(max(nx.connected_components(G), key=len))
        print(f"Graph connectivity:")
        print(f"- Number of connected components: {num_components}")
        print(f"- Size of largest component: {largest_cc}")
        print(f"- Percentage of nodes in largest component: {(largest_cc / graph.num_nodes) * 100:.2f}%")

        # Check edge weight distribution if available
        if hasattr(graph, 'edge_attr'):
            print(f"Edge weight distribution:")
            weights = graph.edge_attr.cpu().numpy()
            print(f"- 25th percentile: {np.percentile(weights, 25):.4f}")
            print(f"- Median: {np.percentile(weights, 50):.4f}")
            print(f"- 75th percentile: {np.percentile(weights, 75):.4f}")
            print(f"- Number of weak edges (< 0.1): {(weights < 0.1).sum()}")
            print(f"- Number of strong edges (> 0.9): {(weights > 0.9).sum()}")

        return graph

    print("\nStarting graph creation/loading process...")
    try:
        # Check if cached graphs exist
        if os.path.exists(cache_file):
            print("\nAttempting to load cached graphs...")
            try:
                with open(cache_file, 'rb') as f:
                    drug_graph, protein_graph = pickle.load(f)
                print("Successfully loaded cached graphs.")

                # Validate loaded graphs (validation happens on CPU)
                print("\nValidating loaded graphs...")
                drug_graph = validate_graph(drug_graph, "drug")
                protein_graph = validate_graph(protein_graph, "protein")

                # Additional validation for cached graphs
                print("\nPerforming additional validation for cached graphs...")
                if not (hasattr(drug_graph, 'x') and hasattr(drug_graph, 'edge_index')):
                    raise ValueError("Cached drug graph missing required attributes")
                if not (hasattr(protein_graph, 'x') and hasattr(protein_graph, 'edge_index')):
                    raise ValueError("Cached protein graph missing required attributes")

                print("\nCache validation successful.")
                return drug_graph, protein_graph

            except Exception as e:
                print(f"\nError loading cached graphs: {str(e)}")
                print("Will regenerate graphs...")

        # Create new graphs
        print("\nCreating new graphs...")
        try:
            print("Calling create_enhanced_graphs...")
            drug_graph, protein_graph = processor.create_enhanced_graphs()
            print("Successfully created new graphs.")

            # Validate new graphs (validation happens on CPU)
            print("\nValidating newly created graphs...")
            drug_graph = validate_graph(drug_graph, "drug")
            protein_graph = validate_graph(protein_graph, "protein")

            # Save the new graphs while on CPU
            print("\nSaving new graphs to cache...")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump((drug_graph, protein_graph), f)
                print("Successfully cached new graphs.")
            except Exception as e:
                print(f"Warning: Failed to cache graphs: {str(e)}")
                print("Continuing without caching...")

            return drug_graph, protein_graph

        except Exception as e:
            print(f"\nError creating new graphs: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    except Exception as e:
        print(f"\nCritical error in graph creation/loading: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Add LayerNorm
            nn.GELU(),
            nn.Dropout(0.1)  # Add dropout for regularization
        )

        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv = GCNConv(hidden_dim, hidden_dim)
            # Initialize conv weights with very small values
            conv.lin.weight.data.uniform_(-0.01, 0.01)
            if conv.lin.bias is not None:
                conv.lin.bias.data.zero_()
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

    def _safe_normalize(self, x, eps=1e-12):
        """Safe normalization with minimum norm check."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=eps)
        return x / norm

    def forward(self, x, edge_index, edge_attr):
        """Forward pass with aggressive value control."""
        # Initial projection
        h = self.input_projection(x)

        # Initial normalization
        h = self._safe_normalize(h)
        h = torch.clamp(h, min=-1.0, max=1.0)

        # Edge weight preprocessing
        edge_attr = torch.clamp(edge_attr, min=0.0, max=1.0)
        edge_attr = F.normalize(edge_attr, p=1, dim=0)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Store previous state
            h_prev = h.clone()  # Explicit cloning

            # Pre-conv normalization
            h = self._safe_normalize(h)

            try:
                # Convolution with safeguards
                h_new = conv(h, edge_index, edge_attr)

                # Immediate post-conv checks
                if torch.isnan(h_new).any() or torch.isinf(h_new).any():
                    print(f"NaN/Inf in layer {i + 1} conv output")
                    h_new = h_prev  # Revert to previous state
                else:
                    # Value control
                    h_new = torch.clamp(h_new, min=-1.0, max=1.0)
                    h_new = self._safe_normalize(h_new)
                    h_new = norm(h_new)
                    h_new = F.gelu(h_new)

                    # Very conservative residual
                    h = h_prev + 0.001 * h_new  # Further reduced scale

                    # Final normalization
                    h = self._safe_normalize(h)
                    h = torch.clamp(h, min=-1.0, max=1.0)

                # Debugging
                # print(f"\nLayer {i + 1} stats:")
                # print(f"h_new range: [{h_new.min():.4f}, {h_new.max():.4f}], mean: {h_new.mean():.4f}")
                # print(f"h range: [{h.min():.4f}, {h.max():.4f}], mean: {h.mean():.4f}")

            except Exception as e:
                print(f"Error in layer {i + 1}: {str(e)}")
                h = h_prev  # Fallback to previous state
                continue

        # Final normalization
        h = self._safe_normalize(h)
        return torch.clamp(h, min=-1.0, max=1.0)


class AttentionFusion(nn.Module):
    def __init__(self, drug_dim, protein_dim):
        super(AttentionFusion, self).__init__()

        combined_dim = drug_dim + protein_dim

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(combined_dim, FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(FUSION_DIM, 1),
            nn.Sigmoid()
        )

        # Main network
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(FUSION_DIM, 1)
        )

    def forward(self, drug_embed, protein_embed):
        try:
            # Concatenate embeddings
            combined = torch.cat([drug_embed, protein_embed], dim=1)

            # Compute attention weights
            attention_weights = self.attention(combined)

            # Apply attention and compute output
            weighted_features = combined * attention_weights
            output = self.fusion(weighted_features)

            return output, attention_weights

        except Exception as e:
            print(f"Error in AttentionFusion forward pass: {str(e)}")
            raise

class EnhancedDataProcessor:
    def __init__(self, train_interactions, test_interactions,
                 drug_features, protein_features,
                 similarity_threshold=0.5,
                 k_neighbors=5):

        self.similarity_threshold = similarity_threshold
        self.k_neighbors = k_neighbors

        # Initialize scalers
        self.drug_scaler = StandardScaler()
        self.protein_scaler = StandardScaler()

        # Process and clean data
        self.process_initial_data(
            train_interactions,
            test_interactions,
            drug_features,
            protein_features
        )

    def _validate_mappings(self):
        """Validate mappings with detailed diagnostics."""
        # Get all entities from filtered interactions
        train_drugs = set(self.train_interactions['chemical'])
        test_drugs = set(self.test_interactions['chemical'])
        all_interaction_drugs = train_drugs | test_drugs

        train_proteins = set(self.train_interactions['protein'])
        test_proteins = set(self.test_interactions['protein'])
        all_interaction_proteins = train_proteins | test_proteins

        # Check for missing mappings
        missing_drug_mappings = all_interaction_drugs - set(self.drug_to_idx.keys())
        missing_protein_mappings = all_interaction_proteins - set(self.protein_to_idx.keys())

        if missing_drug_mappings:
            print("\nWARNING: Found drugs in filtered interactions without mappings!")
            print(f"Number of missing drug mappings: {len(missing_drug_mappings)}")
            print("Sample of problematic drugs:")
            for drug in list(missing_drug_mappings)[:10]:
                print(f"  {drug}")
                if drug in self.valid_drugs:
                    print(f"    Drug is in valid_drugs but not mapped!")
                if drug in self.drug_features:
                    print(f"    Drug has features but not mapped!")
            raise ValueError(f"Found {len(missing_drug_mappings)} drugs in interactions without mappings")

        if missing_protein_mappings:
            print("\nWARNING: Found proteins in filtered interactions without mappings!")
            print(f"Number of missing protein mappings: {len(missing_protein_mappings)}")
            print("Sample of problematic proteins:")
            for protein in list(missing_protein_mappings)[:10]:
                print(f"  {protein}")
            raise ValueError(f"Found {len(missing_protein_mappings)} proteins in interactions without mappings")

    def process_initial_data(self, train_interactions, test_interactions,
                           drug_features, protein_features):
        """Process and clean initial data with detailed diagnostics."""
        print("\nProcessing and cleaning initial data...")

        # Clean features and get valid entities
        print("\nCleaning features...")
        self.drug_features, self.protein_features = self.clean_features(
            drug_features,
            protein_features
        )

        # Get valid entities (those with features) as sets
        self.valid_drugs = set(self.drug_features.keys())
        self.valid_proteins = set(self.protein_features.keys())

        print(f"\nFeature Statistics:")
        print(f"Number of drugs with features: {len(self.valid_drugs)}")
        print(f"Number of proteins with features: {len(self.valid_proteins)}")

        # Analyze interactions before filtering
        all_drugs_in_interactions = set(pd.concat([
            train_interactions['chemical'],
            test_interactions['chemical']
        ]).unique())

        all_proteins_in_interactions = set(pd.concat([
            train_interactions['protein'],
            test_interactions['protein']
        ]).unique())

        print("\nInteraction Statistics (Before Filtering):")
        print(f"Total unique drugs in interactions: {len(all_drugs_in_interactions)}")
        print(f"Total unique proteins in interactions: {len(all_proteins_in_interactions)}")

        # Find missing entities
        missing_drugs = all_drugs_in_interactions - self.valid_drugs
        missing_proteins = all_proteins_in_interactions - self.valid_proteins

        print("\nMissing Entity Analysis:")
        print(f"Drugs in interactions but missing features: {len(missing_drugs)}")
        if missing_drugs:
            print("Sample of missing drugs:")
            for drug in list(missing_drugs)[:10]:
                print(f"  {drug}")

        print(f"\nProteins in interactions but missing features: {len(missing_proteins)}")
        if missing_proteins:
            print("Sample of missing proteins:")
            for protein in list(missing_proteins)[:10]:
                print(f"  {protein}")

        # Filter interactions
        print("\nFiltering interactions...")
        self.train_interactions = train_interactions[
            train_interactions['chemical'].isin(self.valid_drugs) &
            train_interactions['protein'].isin(self.valid_proteins)
        ].copy().reset_index(drop=True)

        self.test_interactions = test_interactions[
            test_interactions['chemical'].isin(self.valid_drugs) &
            test_interactions['protein'].isin(self.valid_proteins)
        ].copy().reset_index(drop=True)

        print("\nInteraction Statistics (After Filtering):")
        print(f"Training interactions: {len(self.train_interactions)}")
        print(f"Testing interactions: {len(self.test_interactions)}")

        # Verify filtering
        remaining_drugs = set(pd.concat([
            self.train_interactions['chemical'],
            self.test_interactions['chemical']
        ]).unique())

        remaining_proteins = set(pd.concat([
            self.train_interactions['protein'],
            self.test_interactions['protein']
        ]).unique())

        print(f"\nRemaining unique entities after filtering:")
        print(f"Drugs: {len(remaining_drugs)}")
        print(f"Proteins: {len(remaining_proteins)}")

        # Create entity mappings
        print("\nCreating entity mappings...")
        self.create_entity_mappings()

        # Final validation
        self._validate_mappings()

    def clean_features(self, drug_features, protein_features):
        """Clean and normalize features."""
        print("\nCleaning and normalizing features...")

        cleaned_drug_features = {}
        cleaned_protein_features = {}

        # Clean drug features
        drug_feature_matrix = []
        valid_drug_ids = []

        for drug_id, (ecfp, desc) in drug_features.items():
            if isinstance(ecfp, np.ndarray) and isinstance(desc, np.ndarray):
                if not (np.isnan(ecfp).any() or np.isnan(desc).any()):
                    combined_features = np.concatenate([ecfp, desc])
                    drug_feature_matrix.append(combined_features)
                    valid_drug_ids.append(drug_id)
                    cleaned_drug_features[drug_id] = combined_features

        # Normalize drug features
        if drug_feature_matrix:
            drug_feature_matrix = np.vstack(drug_feature_matrix)
            normalized_drug_features = self.drug_scaler.fit_transform(drug_feature_matrix)
            for idx, drug_id in enumerate(valid_drug_ids):
                cleaned_drug_features[drug_id] = normalized_drug_features[idx]

        # Clean protein features (similar process)
        protein_feature_matrix = []
        valid_protein_ids = []

        for protein_id, (ecfp, desc) in protein_features.items():
            if isinstance(ecfp, np.ndarray) and isinstance(desc, np.ndarray):
                if not (np.isnan(ecfp).any() or np.isnan(desc).any()):
                    combined_features = np.concatenate([ecfp, desc])
                    protein_feature_matrix.append(combined_features)
                    valid_protein_ids.append(protein_id)
                    cleaned_protein_features[protein_id] = combined_features

        # Normalize protein features
        if protein_feature_matrix:
            protein_feature_matrix = np.vstack(protein_feature_matrix)
            normalized_protein_features = self.protein_scaler.fit_transform(protein_feature_matrix)
            for idx, protein_id in enumerate(valid_protein_ids):
                cleaned_protein_features[protein_id] = normalized_protein_features[idx]

        return cleaned_drug_features, cleaned_protein_features

    def create_entity_mappings(self):
        """Create mappings between entities and indices."""
        # Get unique entities from filtered interactions
        interaction_drugs = set(pd.concat([
            self.train_interactions['chemical'],
            self.test_interactions['chemical']
        ]).unique())

        interaction_proteins = set(pd.concat([
            self.train_interactions['protein'],
            self.test_interactions['protein']
        ]).unique())

        # Ensure we only map entities that are both in interactions and have features
        self.valid_drugs = sorted(list(interaction_drugs & self.valid_drugs))
        self.valid_proteins = sorted(list(interaction_proteins & self.valid_proteins))

        # Create mappings
        self.drug_to_idx = {drug: idx for idx, drug in enumerate(self.valid_drugs)}
        self.protein_to_idx = {protein: idx for idx, protein in enumerate(self.valid_proteins)}

        print("\nMapping Statistics:")
        print(f"Drugs mapped: {len(self.drug_to_idx)}")
        print(f"Proteins mapped: {len(self.protein_to_idx)}")

    def create_enhanced_graphs(self):
        """Create simplified graph structures for drugs and proteins."""
        print("\nCreating graph structures...")

        # Load or generate similarities
        drug_sim_df, protein_sim_df = self._load_or_generate_similarities(
            covered_drugs=set(self.valid_drugs),
            covered_proteins=set(self.valid_proteins),
            drug_sim_file='drug_similarities.csv',
            protein_sim_file='protein_similarities.csv',
            subset_size=5
        )

        # Convert similarity DataFrames to matrices
        drug_sim_matrix = self._convert_similarities_to_matrix(
            drug_sim_df,
            self.valid_drugs,
            'drug'
        )

        protein_sim_matrix = self._convert_similarities_to_matrix(
            protein_sim_df,
            self.valid_proteins,
            'protein'
        )

        # Create basic graphs
        drug_graph = self.create_single_graph(
            drug_sim_matrix,
            self.drug_features,
            self.valid_drugs,
            'drug'
        )

        protein_graph = self.create_single_graph(
            protein_sim_matrix,
            self.protein_features,
            self.valid_proteins,
            'protein'
        )

        return drug_graph, protein_graph

    def create_single_graph(self, similarity_matrix, features, valid_entities, entity_type):
        """Create a single graph structure from similarity matrix."""
        print(f"\nCreating {entity_type} graph...")

        # Convert similarity matrix to edges and weights  - Debuging
        # print("Similarity matrix stats:")
        # print(f"- Shape: {similarity_matrix.shape}")
        # print(f"- Min: {similarity_matrix.min()}")
        # print(f"- Max: {similarity_matrix.max()}")
        # print(f"- Mean: {similarity_matrix.mean()}")
        # print(f"- NaN count: {np.isnan(similarity_matrix).sum()}")

        # Convert similarity matrix to edges and weights
        edges = []
        edge_weights = []
        n = len(valid_entities)

        # Get non-zero similarities (excluding self-loops)
        rows, cols = np.nonzero(similarity_matrix)
        for i, j in zip(rows, cols):
            if i != j:  # Exclude self-loops for now
                weight = similarity_matrix[i, j]
                # Validate and clean the weight
                if np.isnan(weight) or np.isinf(weight):
                    print(f"Warning: Found {weight} weight between nodes {i} and {j}")
                    continue
                # Clamp weight to valid range
                weight = float(np.clip(weight, 0.0, 1.0))
                edges.append([i, j])
                edge_weights.append(weight)

        # Add self-loops with weight 1.0
        for i in range(n):
            edges.append([i, i])
            edge_weights.append(1.0)

        # Create edge index and edge attributes
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        else:
            # Fallback to only self-loops if no edges found
            edge_index = torch.arange(n).repeat(2, 1)
            edge_attr = torch.ones(n)

        # Create node features - directly use the features array
        x = torch.tensor(
            [features[entity] for entity in valid_entities],
            dtype=torch.float
        )

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        print(f"Created graph with {data.num_nodes} nodes and {data.num_edges} edges")
        print(f"Node feature dimension: {data.num_node_features}")

        # Additional statistics
        unique_weights = torch.unique(edge_attr)
        print(f"Edge weight statistics:")
        print(f"  Min weight: {edge_attr.min():.4f}")
        print(f"  Max weight: {edge_attr.max():.4f}")
        print(f"  Mean weight: {edge_attr.mean():.4f}")
        print(f"  Unique weights: {len(unique_weights)}")

        return data

    def _convert_similarities_to_matrix(self, sim_df, entities, entity_type):
        """Convert similarity DataFrame to matrix format with entity filtering."""
        n = len(entities)
        entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}
        sim_matrix = np.zeros((n, n))
        np.fill_diagonal(sim_matrix, 1.0)  # Self-loops

        # Filter similarity DataFrame to only include valid entities
        if entity_type == 'drug':
            filtered_sim_df = sim_df[
                sim_df['drug1'].isin(entities) &
                sim_df['drug2'].isin(entities)
                ]
            entity1_col = 'drug1'
            entity2_col = 'drug2'
        else:
            filtered_sim_df = sim_df[
                sim_df['protein1'].isin(entities) &
                sim_df['protein2'].isin(entities)
                ]
            entity1_col = 'protein1'
            entity2_col = 'protein2'

        # Fill similarity matrix using filtered data
        for _, row in filtered_sim_df.iterrows():
            idx1 = entity_to_idx[row[entity1_col]]
            idx2 = entity_to_idx[row[entity2_col]]
            sim_matrix[idx1, idx2] = row['similarity_score']
            sim_matrix[idx2, idx1] = row['similarity_score']

        return sim_matrix

    def _load_or_generate_similarities(self, covered_drugs, covered_proteins,
                                       drug_sim_file, protein_sim_file, subset_size):
        """Load existing or generate new similarities using covered entities."""

        # Check if similarity files exist and are complete
        if os.path.exists(drug_sim_file) and os.path.exists(protein_sim_file):
            print("\nChecking existing similarity files...")
            drug_similarities = pd.read_csv(drug_sim_file)
            protein_similarities = pd.read_csv(protein_sim_file)

            drug_coverage = covered_drugs.issubset(
                set(drug_similarities['drug1']).union(set(drug_similarities['drug2']))
            )
            protein_coverage = covered_proteins.issubset(
                set(protein_similarities['protein1']).union(set(protein_similarities['protein2']))
            )

            if drug_coverage and protein_coverage:
                print("Existing similarity files are complete and valid.")
                return drug_similarities, protein_similarities

        print("\nGenerating similarity matrices for covered entities...")

        # Compute cosine similarity directly
        def compute_similarity(feature1, feature2):
            if isinstance(feature1, np.ndarray) and isinstance(feature2, np.ndarray):
                return np.dot(feature1, feature2) / (
                        np.linalg.norm(feature1) * np.linalg.norm(feature2)
                )
            return None

        # Generate similarities only for covered entities
        drug_similarities = {}
        for drug1 in covered_drugs:
            random_subset = random.sample(list(covered_drugs), min(subset_size, len(covered_drugs)))
            for drug2 in random_subset:
                if drug1 != drug2:
                    similarity = compute_similarity(
                        self.drug_features[drug1],
                        self.drug_features[drug2]
                    )
                    if similarity is not None:
                        drug_similarities[(drug1, drug2)] = similarity

        protein_similarities = {}
        for protein1 in covered_proteins:
            random_subset = random.sample(list(covered_proteins), min(subset_size, len(covered_proteins)))
            for protein2 in random_subset:
                if protein1 != protein2:
                    similarity = compute_similarity(
                        self.protein_features[protein1],
                        self.protein_features[protein2]
                    )
                    if similarity is not None:
                        protein_similarities[(protein1, protein2)] = similarity

        # Convert to DataFrames
        drug_sim_df = pd.DataFrame([
            {'drug1': d1, 'drug2': d2, 'similarity_score': score}
            for (d1, d2), score in drug_similarities.items()
        ])

        protein_sim_df = pd.DataFrame([
            {'protein1': p1, 'protein2': p2, 'similarity_score': score}
            for (p1, p2), score in protein_similarities.items()
        ])

        # Save to files
        drug_sim_df.to_csv(drug_sim_file, index=False)
        protein_sim_df.to_csv(protein_sim_file, index=False)

        print(f"\nGenerated and saved similarity matrices:")
        print(f"Drug similarities: {len(drug_similarities)} pairs")
        print(f"Protein similarities: {len(protein_similarities)} pairs")

        return drug_sim_df, protein_sim_df

    def create_enhanced_dataset(self, interactions, drug_graph, protein_graph, rf_predictions, device):
        """Create an enhanced dataset with additional features."""
        return EnhancedInteractionDataset(
            interactions=interactions,
            drug_graph_data=drug_graph,
            protein_graph_data=protein_graph,
            drug_to_idx=self.drug_to_idx,
            protein_to_idx=self.protein_to_idx,
            rf_predictions=rf_predictions,
            device=device
        )

class EnhancedInteractionDataset(torch.utils.data.Dataset):
    def __init__(self, interactions, drug_graph_data, protein_graph_data,
                 drug_to_idx, protein_to_idx, rf_predictions, device='cuda:0'):
        """Initialize dataset with drug filtering."""
        print("\nInitializing dataset with drug filtering...")
        self.k_hops = 2  # k_hops
        self.num_neighbors = 5  # num_neighbors
        # Store original length
        original_len = len(interactions)

        # Find all valid interactions
        valid_mask = interactions['chemical'].isin(drug_to_idx.keys()) & \
                     interactions['protein'].isin(protein_to_idx.keys())

        # Filter interactions and reset index
        self.interactions = interactions[valid_mask].reset_index(drop=True)

        # Filter RF predictions to match
        self.rf_predictions = torch.tensor(rf_predictions[valid_mask], dtype=torch.float)

        # Store other attributes
        self.drug_graph_data = drug_graph_data
        self.protein_graph_data = protein_graph_data
        self.drug_to_idx = drug_to_idx
        self.protein_to_idx = protein_to_idx
        self.device = device

        # Print filtering statistics
        removed_count = original_len - len(self.interactions)
        print(f"\nDataset filtering summary:")
        print(f"Original interactions: {original_len}")
        print(f"Filtered interactions: {len(self.interactions)}")
        print(f"Removed interactions: {removed_count}")

        if removed_count > 0:
            # Show examples of removed drugs
            removed_drugs = set(interactions[~valid_mask]['chemical'].unique())
            print("\nExamples of removed drugs:")
            for drug in list(removed_drugs)[:5]:
                print(f"  {drug}")

        # Move data to device
        self.drug_graph_data = self._to_device(drug_graph_data)
        self.protein_graph_data = self._to_device(protein_graph_data)
        self.rf_predictions = self.rf_predictions.to(device)

        # Create adjacency matrices
        self.drug_adj = self._create_adjacency_matrix(self.drug_graph_data)
        self.protein_adj = self._create_adjacency_matrix(self.protein_graph_data)

        # Pre-compute neighborhoods
        self.drug_neighborhoods = {}
        self.protein_neighborhoods = {}
        self._precompute_neighborhoods()

    def _validate_data(self):
        """Validate that all required mappings exist."""
        missing_drugs = []
        missing_proteins = []

        for idx, row in self.interactions.iterrows():
            if row['chemical'] not in self.drug_to_idx:
                missing_drugs.append((idx, row['chemical']))
            if row['protein'] not in self.protein_to_idx:
                missing_proteins.append((idx, row['protein']))

        if missing_drugs:
            print("\nMissing drug mappings:")
            for idx, drug in missing_drugs[:5]:
                print(f"Row {idx}: {drug}")
            raise ValueError(f"Found {len(missing_drugs)} drugs without mappings")

        if missing_proteins:
            print("\nMissing protein mappings:")
            for idx, protein in missing_proteins[:5]:
                print(f"Row {idx}: {protein}")
            raise ValueError(f"Found {len(missing_proteins)} proteins without mappings")


    def _to_device(self, data):
        """Move PyG Data object to device."""
        return Data(
            x=data.x.to(self.device),
            edge_index=data.edge_index.to(self.device),
            edge_attr=data.edge_attr.to(self.device)
        )

    def _create_adjacency_matrix(self, graph_data):
        """Create dense adjacency matrix for faster neighborhood computation."""
        num_nodes = graph_data.num_nodes
        adj = torch.zeros((num_nodes, num_nodes), device=self.device)
        adj[graph_data.edge_index[0], graph_data.edge_index[1]] = graph_data.edge_attr
        return adj

    def _get_k_hop_neighbors(self, adj, center_idx, k_hops):
        """Get k-hop neighborhood indices using adjacency matrix."""
        neighbors = {center_idx}
        current_neighbors = {center_idx}
        device = self.device
        for _ in range(k_hops):
            next_neighbors = set()
            for node in current_neighbors:
                # Get connected nodes
                #adj = adj.to(device)
                connected = torch.where(adj[node] > 0)[0].tolist()
                next_neighbors.update(connected)
            current_neighbors = next_neighbors - neighbors
            neighbors.update(current_neighbors)

            # Limit size of neighborhood
            if len(neighbors) > self.num_neighbors:
                neighbors = set(list(neighbors)[:self.num_neighbors])
                break

        return sorted(list(neighbors))

    def _precompute_neighborhoods(self):
        """Pre-compute and cache all k-hop neighborhoods."""
        print("Pre-computing neighborhoods...")

        # Drug neighborhoods
        for drug_idx in range(self.drug_graph_data.num_nodes):
            neighbor_indices = self._get_k_hop_neighbors(
                self.drug_adj,
                drug_idx,
                self.k_hops
            )
            self.drug_neighborhoods[drug_idx] = neighbor_indices

        # Protein neighborhoods
        for protein_idx in range(self.protein_graph_data.num_nodes):
            neighbor_indices = self._get_k_hop_neighbors(
                self.protein_adj,
                protein_idx,
                self.k_hops
            )
            self.protein_neighborhoods[protein_idx] = neighbor_indices

    def __len__(self):
        """Return the number of interactions."""
        return len(self.interactions)

    def __getitem__(self, idx):
        """Get a single interaction with indices."""
        interaction = self.interactions.iloc[idx]

        # Get indices
        drug_idx = self.drug_to_idx[interaction['chemical']]
        protein_idx = self.protein_to_idx[interaction['protein']]

        # Get RF prediction and label
        rf_pred = self.rf_predictions[idx]
        label = torch.tensor(interaction['label'], dtype=torch.float)

        return drug_idx, protein_idx, rf_pred, label

    def get_full_graphs(self):
        """Return the full graph data."""
        return self.drug_graph_data, self.protein_graph_data

    def compute_statistics(self):
        """Compute and store dataset statistics."""
        self.statistics = {
            'num_interactions': len(self.interactions),
            'num_drugs': len(set(self.interactions['chemical'])),
            'num_proteins': len(set(self.interactions['protein'])),
            'num_positive': int(self.interactions['label'].sum()),
            'num_negative': int(len(self.interactions) - self.interactions['label'].sum()),
            'drug_graph_nodes': self.drug_graph_data.num_nodes,
            'drug_graph_edges': self.drug_graph_data.num_edges,
            'protein_graph_nodes': self.protein_graph_data.num_nodes,
            'protein_graph_edges': self.protein_graph_data.num_edges
        }

        # Compute class weights for balanced training
        pos_weight = (len(self.interactions) - self.statistics['num_positive']) / self.statistics['num_positive']
        self.statistics['pos_weight'] = pos_weight

        print("\nDataset Statistics:")
        for key, value in self.statistics.items():
            print(f"{key}: {value}")

    def get_pos_weight(self):
        """Return positive class weight for balanced training."""
        return self.statistics['pos_weight']

    def get_statistics(self):
        """Return dataset statistics."""
        return self.statistics


def evaluate_predictions(true_labels, predictions):
    """Compute comprehensive evaluation metrics."""
    predictions_binary = (predictions >= 0.5).astype(float)

    return {
        'accuracy': accuracy_score(true_labels, predictions_binary),
        'precision': precision_score(true_labels, predictions_binary),
        'recall': recall_score(true_labels, predictions_binary),
        'f1': f1_score(true_labels, predictions_binary),
        'auc_roc': roc_auc_score(true_labels, predictions),
        'auprc': average_precision_score(true_labels, predictions)
    }


def collate_batch(batch):
    """Collate indices into batches."""
    drug_indices = []
    protein_indices = []
    rf_preds = []
    labels = []

    for drug_idx, protein_idx, rf_pred, label in batch:
        drug_indices.append(drug_idx)
        protein_indices.append(protein_idx)
        rf_preds.append(rf_pred)
        labels.append(label)

    return (
        torch.tensor(drug_indices, dtype=torch.long),
        torch.tensor(protein_indices, dtype=torch.long),
        torch.tensor(rf_preds, dtype=torch.float),
        torch.tensor(labels, dtype=torch.float)
    )


def main():
    """Main function restructured with full graph attention approach."""
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True

    # Set random seed for reproducibility
    set_seed(SEED)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    print("\nLoading data...")
    train_interactions = pd.read_csv('../Data/StitchString/cv_0/stitch-data-tr.csv')
    test_interactions = pd.read_csv('../Data/StitchString/cv_0/stitch-data-ts.csv')

    with open('../simgraphmaker/features.pkl', 'rb') as f:
        features = pickle.load(f)

    print("Data loaded successfully!")
    print(f"Training interactions: {len(train_interactions)}")
    print(f"Testing interactions: {len(test_interactions)}")

    # Random Forest Phase
    print("\n" + "=" * 50)
    print("Phase 1: Random Forest Baseline")
    print("=" * 50)

    rf_model = RandomForestBaseline(n_estimators=500)

    try:
        # Train and get RF predictions
        print("\nTraining Random Forest model...")
        train_rf_preds = rf_model.fit(
            train_interactions,
            features['drug_features'],
            features['protein_features']
        )

        # Get test predictions
        print("\nGenerating test predictions...")
        test_rf_preds = rf_model.predict(
            test_interactions,
            features['drug_features'],
            features['protein_features']
        )

        # Evaluate RF performance
        rf_metrics = evaluate_predictions(
            test_interactions['label'].values,
            test_rf_preds
        )

        print("\nRandom Forest Performance:")
        for metric, value in rf_metrics.items():
            print(f"{metric}: {value:.4f}")

        if rf_metrics['auc_roc'] < 0.5:
            print("\nWarning: Random Forest performance is poor. Please check the data and features.")
            return None, rf_metrics

    except Exception as e:
        print(f"\nError in Random Forest training: {str(e)}")
        return None, None

    # GNN Phase
    print("\n" + "=" * 50)
    print("Phase 2: Enhanced GNN Model")
    print("=" * 50)

    try:
        # Initialize enhanced data processor
        print("\nInitializing data processor...")
        processor = EnhancedDataProcessor(
            train_interactions=train_interactions,
            test_interactions=test_interactions,
            drug_features=features['drug_features'],
            protein_features=features['protein_features'],
            similarity_threshold=0.5,
            k_neighbors=5
        )

        # Create graphs
        drug_graph, protein_graph = create_or_load_graphs(processor)
        # Move to GPU after creation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        drug_graph = drug_graph.to(device)
        protein_graph = protein_graph.to(device)

        # Create datasets
        train_dataset = EnhancedInteractionDataset(
            interactions=train_interactions,
            drug_graph_data=drug_graph,
            protein_graph_data=protein_graph,
            drug_to_idx=processor.drug_to_idx,
            protein_to_idx=processor.protein_to_idx,
            rf_predictions=train_rf_preds,
            device=device
        )

        test_dataset = EnhancedInteractionDataset(
            interactions=test_interactions,
            drug_graph_data=drug_graph,
            protein_graph_data=protein_graph,
            drug_to_idx=processor.drug_to_idx,
            protein_to_idx=processor.protein_to_idx,
            rf_predictions=test_rf_preds,
            device=device
        )

        # Get full graphs for model
        drug_graph_data, protein_graph_data = train_dataset.get_full_graphs()

        # Create train-validation split
        total_size = len(train_dataset)
        train_size = int(0.70 * total_size)
        valid_size = total_size - train_size

        train_subset, valid_subset = torch.utils.data.random_split(
            train_dataset,
            [train_size, valid_size],
            generator=torch.Generator().manual_seed(SEED)
        )

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_batch,
            num_workers=4,
            persistent_workers=True
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=4,
            persistent_workers=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=4,
            persistent_workers=True
        )

        # Initialize model
        model = EnhancedGNNModel(
            drug_input_dim=drug_graph_data.num_node_features,
            protein_input_dim=protein_graph_data.num_node_features
        ).to(device)

        # Move full graphs to device
        drug_graph_data = drug_graph_data.to(device)
        protein_graph_data = protein_graph_data.to(device)

        # Initialize trainer
        trainer = CurriculumTrainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            device=device,
            max_iterations=10,
            lr=LEARNING_RATE,
            warmup_epochs=5,
            checkpoint_dir='checkpoints',
            drug_graph_data=drug_graph_data,
            protein_graph_data=protein_graph_data
        )

        # Train model
        print("\nStarting GNN training...")
        final_metrics = trainer.train(patience=EARLY_STOPPING_PATIENCE)

        print("\nFinal Results:")
        for phase in ['validation', 'test']:
            if final_metrics[phase]:
                print(f"\n{phase.capitalize()} Metrics:")
                for k, v in final_metrics[phase].items():
                    if isinstance(v, float):
                        print(f"{k}: {v:.4f}")

        return model, final_metrics

    except Exception as e:
        print(f"\nError in GNN processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, {'rf_metrics': rf_metrics, 'gnn_metrics': None}


if __name__ == "__main__":
    model, results = main()
