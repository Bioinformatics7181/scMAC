# -------------------------------------------------------------------------
# Name: mapping_unified.py
# Intro: Integrated CV engine for MAP, CGCL, and TCRD.
#        Features: Global Hyperparameter Hub, Config Export, Dual-Track Logging.
# -------------------------------------------------------------------------

import pandas as pd
import numpy as np
import torch
import os
import sys
import shutil
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader

import utils
from network import scMACMap, scMAC_CGCL, TCRD, MIST_TCRb_LLM

# ==========================================
# 0. Global Hyperparameter Hub
# ==========================================
# --- Basic Training ---
BATCH_SIZE = 256
EPOCHS = 1000
LR = 0.001
FILTERS_N = 1
SEED = 2026
LOG_INR = 10               # Shared log interval
AAINDEX_FILE = "resources/AAidx_PCA.txt"

# --- Core Mode Parameters ---
IS_TRAINING_MODE = False   # True: Use val set for model selection; False: Standard logic
USE_V = True               # Toggle V gene feature
USE_J = True               # Toggle J gene feature
IS_UNKNOWN = False         # Enable Strict Clonotype Isolation
CLEAN_MODELS = False       # Delete fold models after execution
ENABLE_REJECTION = True    # Enable Open-Set uncertainty rejection
REJECTION_PERCENTILE = 99.9

# --- Model-Specific Hyperparameters ---
# MAP
MAP_W_ALIGN = 0.05
MAP_W_PROTO = 0.2
MAP_W_KLD = 0.001
# CGCL
CGCL_W_CONTRAST = 1.0
CGCL_W_KLD = 0.01
CGCL_TEMP = 0.1
# TCRD
TCRD_DROPOUT = 0.4
# MIST
MIST_W_MMD = 1.0     # The weight for MMD alignment loss
MIST_W_KL = 0.1      # The weight for VAE KL divergence loss

# ==========================================
# 0.5. Dual-Track Logger
# ==========================================
class Logger(object):
    """ Record console output to a log file while printing to terminal. """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ==========================================
# 1. Unified Lazy Dataset
# ==========================================
class LazyTCRDatasetUnified(Dataset):
    def __init__(self, df_data, npz_matrix, aa_dict, bc_to_matrix_idx, 
                 model_type='map', max_len=24, latent_dim=128, 
                 use_v=True, use_j=True, is_train=True):
        self.df = df_data.reset_index(drop=True)
        self.aa_dict = aa_dict
        self.model_type = model_type.lower()
        self.max_len = max_len
        self.latent_dim = latent_dim
        self.use_v = use_v
        self.use_j = use_j
        self.is_train = is_train
        self.npz_matrix = npz_matrix
        self.bc_to_matrix_idx = bc_to_matrix_idx

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row['CDR3'])
        
        padded = (seq + "X" * self.max_len)[:self.max_len]
        tcr_matrix = [self.aa_dict.get(aa.upper(), [0.0]*15) for aa in padded]
        cdr3_tensor = torch.FloatTensor(tcr_matrix).transpose(0, 1) 
        
        v_idx = torch.tensor(row['V_idx'], dtype=torch.long) if self.use_v else torch.tensor(0)
        j_idx = torch.tensor(row['J_idx'], dtype=torch.long) if self.use_j else torch.tensor(0)
        label = torch.tensor(row['Label_idx'], dtype=torch.long)
        
        if self.is_train and self.npz_matrix is not None:
            matrix_idx = self.bc_to_matrix_idx[row['Barcodes']]
            latent = torch.FloatTensor(np.array(self.npz_matrix[matrix_idx]))
        else:
            latent = torch.zeros(self.latent_dim)

        # Dynamic return: CGCL requires TCR_id for clonal mask
        if self.model_type in ['cgcl', 'mist']:
            tcr_id = torch.tensor(row['TCR_id'], dtype=torch.long)
            return cdr3_tensor, v_idx, j_idx, latent, label, tcr_id
        else:
            return cdr3_tensor, v_idx, j_idx, latent, label

# ==========================================
# 2. Composition Evaluation Helper
# ==========================================
def evaluate_fold_composition(y_true, y_pred, inv_dict, all_indices):
    total = len(y_true)
    results = []
    for l_idx in all_indices:
        actual_p = (y_true == l_idx).sum() / total * 100
        pred_p = (y_pred == l_idx).sum() / total * 100
        results.append({"Cell_Type": inv_dict.get(l_idx, "Unknown"), "Actual_%": actual_p, "Predicted_%": pred_p, "Abs_Err": abs(actual_p - pred_p)})
    
    comp_df = pd.DataFrame(results)
    r_val, _ = pearsonr(comp_df['Actual_%'], comp_df['Predicted_%']) if len(comp_df) > 1 else (0, 0)
    print(f"\n[*] Fold Composition: MAE = {comp_df['Abs_Err'].mean():.2f}%, Pearson r = {r_val:.4f}")
    return comp_df

def calculate_prototypes_from_npz(df_train, npz_matrix, bc_to_matrix_idx, class_num, latent_dim=128):
    """
    Calculate prototypes for known classes. Class 0 (Unknown) remains a zero vector.
    """
    prototypes = np.zeros((class_num, latent_dim), dtype=np.float32)
    for c in range(1, class_num):
        class_bcs = df_train[df_train['Label_idx'] == c]['Barcodes'].values
        valid_indices = [bc_to_matrix_idx[bc] for bc in class_bcs if bc in bc_to_matrix_idx]
        if valid_indices:
            class_vectors = npz_matrix[valid_indices]
            prototypes[c] = np.mean(class_vectors, axis=0)
    return torch.FloatTensor(prototypes)

# ==========================================
# 3. Core CV Engine
# ==========================================
def run_unified_cv(model_type='map', csv_file="tcrb_labels.csv", npz_file="scMACLLM_data.npz", output_dir="Exp_Unified", aaindex_file=None):
    aaindex_file = aaindex_file or AAINDEX_FILE
    model_type = model_type.lower()
    os.makedirs(output_dir, exist_ok=True)
    
    # Enable file logging
    log_file = os.path.join(output_dir, f"training_{model_type}.log")
    sys.stdout = Logger(log_file)
    print(f"=== Initializing CV Engine for {model_type.upper()} ===")

    # Export Configuration parameters
    config_params = {
        k: v for k, v in globals().items() 
        if k.isupper() and not k.startswith('_') and isinstance(v, (int, float, str, bool, list, dict, type(None)))
    }
    config_file = os.path.join(output_dir, f"config_{model_type}.json")
    with open(config_file, "w") as f:
        json.dump(config_params, f, indent=4)
    print(f"[*] Hyperparameters locked and exported to '{config_file}'")

    utils.seed_torch(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Hardware detected: Device = {device}")
    
    # --- Load Data & Vocabulary Construction ---
    df = pd.read_csv(csv_file).dropna(subset=['CDR3'])
    aa_dict = utils.get_features(aaindex_file)
    npz_matrix, bc_to_matrix_idx = None, {}

    if model_type in ['map', 'cgcl', 'mist']:
        npz_data = np.load(npz_file, mmap_mode='r', allow_pickle=True)
        npz_matrix, barcodes_array = npz_data['matrix'], npz_data['barcodes']

        # Optional zero-vector ablation.
        # npz_matrix = np.zeros_like(npz_matrix)

        global LATENT_DIM
        LATENT_DIM = npz_matrix.shape[1]
        print(f"[*] Dynamically detected LATENT_DIM from NPZ: {LATENT_DIM}")
        bc_to_matrix_idx = { (bc.decode() if isinstance(bc, bytes) else bc): idx for idx, bc in enumerate(barcodes_array)}
        df = df[df['Barcodes'].isin(bc_to_matrix_idx.keys())].reset_index(drop=True)

    def build_global_dict(series):
        unique_vals = sorted(series.dropna().unique())
        return {val: idx + 1 for idx, val in enumerate(unique_vals)}

    unique_funcs = sorted(df['Function'].dropna().unique())
    if model_type == 'tcrd':
        label_dict = {val: idx for idx, val in enumerate(unique_funcs)}
        label_vocab_size = len(label_dict)
    else:
        label_dict = {val: idx + 1 for idx, val in enumerate(unique_funcs)}
        label_vocab_size = len(label_dict) + 1

    v_dict = build_global_dict(df['V']) if USE_V else {}
    j_dict = build_global_dict(df['J']) if USE_J else {}
    
    vocab_data = {
        "label_dict": label_dict, "v_dict": v_dict, "j_dict": j_dict,
        "label_vocab_size": label_vocab_size,
        "v_vocab_size": len(v_dict) + 1, "j_vocab_size": len(j_dict) + 1
    }
    
    with open(os.path.join(output_dir, f"vocab_{model_type}.json"), "w") as f: 
        json.dump(vocab_data, f, indent=4)

    df['Label_idx'] = df['Function'].map(label_dict).fillna(0).astype(int)
    df['V_idx'] = df['V'].map(v_dict).fillna(0).astype(int) if USE_V else 0
    df['J_idx'] = df['J'].map(j_dict).fillna(0).astype(int) if USE_J else 0
    df['TCR_id'] = df['CDR3'].map({v: i for i, v in enumerate(df['CDR3'].unique())})

    splits = utils.get_mapping_splits(df, n_splits=5, is_unknown=IS_UNKNOWN, seed=SEED)
    
    overall_true, overall_pred, overall_barcodes = [], [], []
    global_best_fold_f1 = -1.0
    final_model_name = os.path.join(output_dir, f"final_model_{model_type}.pth")
    thresholds_record = {}

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n{'='*50}\n         Starting Fold {fold + 1} ({model_type.upper()})\n{'='*50}")
        train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        
        model_save_path = os.path.join(output_dir, f"fold_{fold + 1}_{model_type}.pth")
        
        if model_type in ['map', 'cgcl', 'mist']:
            if model_type == 'mist':
                max_len = 30
            else:
                max_len = 24

            train_ds = LazyTCRDatasetUnified(train_df, npz_matrix, aa_dict, bc_to_matrix_idx, model_type=model_type, max_len=max_len, use_v=USE_V, use_j=USE_J, is_train=True)
            test_ds = LazyTCRDatasetUnified(test_df, npz_matrix, aa_dict, bc_to_matrix_idx, model_type=model_type, max_len=max_len, use_v=USE_V, use_j=USE_J, is_train=False)
            
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
            train_eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

        # --- Phase 1: Training ---
        if model_type in ['map', 'cgcl', 'mist']:
            c_counts = np.bincount(train_df['Label_idx'], minlength=vocab_data['label_vocab_size'])
            c_weights = np.zeros(vocab_data['label_vocab_size'], dtype=np.float32)
            valid_mask = c_counts > 0
            c_weights[valid_mask] = 1.0 / (np.sqrt(c_counts[valid_mask]) + 1e-6)
            c_weights = c_weights / np.sum(c_weights) * (vocab_data['label_vocab_size'] - 1)
            
            if model_type == 'map':
                protos = calculate_prototypes_from_npz(train_df, npz_matrix, bc_to_matrix_idx, vocab_data['label_vocab_size'], latent_dim=LATENT_DIM)
                fold_best_f1 = scMACMap.training(train_loader, test_loader, train_eval_loader, train_df['Label_idx'].values, test_df['Label_idx'].values, protos, c_weights, vocab_data['v_vocab_size'], vocab_data['j_vocab_size'], LATENT_DIM, FILTERS_N, vocab_data['label_vocab_size'], LR, EPOCHS, model_save_path, device, w_align=MAP_W_ALIGN, w_proto=MAP_W_PROTO, w_kld=MAP_W_KLD, log_inr=LOG_INR, is_training_mode=IS_TRAINING_MODE)
            elif model_type == 'cgcl':
                fold_best_f1 = scMAC_CGCL.training(train_loader, test_loader, train_eval_loader, train_df['Label_idx'].values, test_df['Label_idx'].values, c_weights, vocab_data['v_vocab_size'], vocab_data['j_vocab_size'], LATENT_DIM, FILTERS_N, vocab_data['label_vocab_size'], LR, EPOCHS, model_save_path, device, w_contrast=CGCL_W_CONTRAST, w_kld=CGCL_W_KLD, temp=CGCL_TEMP, log_inr=LOG_INR, is_training_mode=IS_TRAINING_MODE)
            elif model_type == 'mist':
                # Call MIST training interface
                fold_best_f1 = MIST_TCRb_LLM.train_network(
                    train_loader=train_loader, 
                    test_loader=test_loader, 
                    v_vocab_size=vocab_data['v_vocab_size'], 
                    j_vocab_size=vocab_data['j_vocab_size'], 
                    llm_dim=LATENT_DIM, 
                    class_num=vocab_data['label_vocab_size'], 
                    epochs=EPOCHS, 
                    lr=LR, 
                    is_training_mode=IS_TRAINING_MODE, 
                    model_f=model_save_path, 
                    device=device, 
                    w_mmd=MIST_W_MMD, 
                    w_kl=MIST_W_KL
                )

        elif model_type == 'tcrd':
            fold_best_f1 = TCRD.training(train_df['CDR3'].tolist(), train_df['Label_idx'].tolist(), test_df['CDR3'].tolist(), test_df['Label_idx'].tolist(), FILTERS_N, vocab_data['label_vocab_size'], LR, EPOCHS, TCRD_DROPOUT, LOG_INR, model_save_path, aaindex_file, device, IS_TRAINING_MODE, BATCH_SIZE)

        # --- Phase 2: Calibration (Adaptive Threshold) ---
        adaptive_threshold = None
        if model_type in ['map', 'cgcl', 'mist']:
            if model_type == 'map': net = scMACMap
            elif model_type == 'cgcl': net = scMAC_CGCL
            elif model_type == 'mist': net = MIST_TCRb_LLM

            if model_type == 'mist':
                _, train_uncertainties = net.prediction(train_eval_loader, vocab_data['v_vocab_size'], vocab_data['j_vocab_size'], LATENT_DIM, vocab_data['label_vocab_size'], model_save_path, device)
            else:
                _, train_uncertainties = net.prediction(train_eval_loader, vocab_data['v_vocab_size'], vocab_data['j_vocab_size'], LATENT_DIM, FILTERS_N, vocab_data['label_vocab_size'], model_save_path, device)

            adaptive_threshold = float(np.percentile(train_uncertainties, REJECTION_PERCENTILE))
            
            thresholds_record[f"fold_{fold + 1}"] = adaptive_threshold
            print(f"    -> Adaptive Threshold recorded: {adaptive_threshold:.4f}")

        if IS_TRAINING_MODE:
            if fold_best_f1 > global_best_fold_f1:
                global_best_fold_f1 = fold_best_f1
                shutil.copy(model_save_path, final_model_name)
                print(f"    [Checkpoint] Fold {fold + 1} produced best model so far (F1: {fold_best_f1:.4f})")
                if adaptive_threshold is not None:
                    thresholds_record["final_model"] = adaptive_threshold

        # --- Phase 3: Open-Set Inference & Evaluation ---
        if model_type in ['map', 'cgcl', 'mist']:
            if model_type == 'mist':
                probs, test_uncertainties = net.prediction(test_loader, vocab_data['v_vocab_size'], vocab_data['j_vocab_size'], LATENT_DIM, vocab_data['label_vocab_size'], model_save_path, device)
            else:
                probs, test_uncertainties = net.prediction(test_loader, vocab_data['v_vocab_size'], vocab_data['j_vocab_size'], LATENT_DIM, FILTERS_N, vocab_data['label_vocab_size'], model_save_path, device)

            preds = np.argmax(np.array(probs)[:, 1:], axis=1) + 1
            
            if ENABLE_REJECTION:
                rejected_mask = np.array(test_uncertainties) > adaptive_threshold
                preds[rejected_mask] = 0
                print(f"    -> Intercepted {np.sum(rejected_mask)} highly uncertain TCRs.")
        
        elif model_type == 'tcrd':
            probs = TCRD.prediction(test_df['CDR3'].tolist(), FILTERS_N, vocab_data['label_vocab_size'], model_save_path, aaindex_file, device)
            preds = np.argmax(probs, axis=1)

        overall_true.extend(test_df['Label_idx'].values)
        overall_pred.extend(preds)
        overall_barcodes.extend(test_df['Barcodes'].values if 'Barcodes' in test_df.columns else np.arange(len(test_df)))
        
        inv_label_dict = {v: k for k, v in label_dict.items()}
        if model_type != 'tcrd': inv_label_dict[0] = "Unknown"
        all_indices = sorted(label_dict.values()) + ([0] if model_type != 'tcrd' else [])
        
        print(f"\n[Fold {fold+1} Seq Report]")
        print(classification_report(test_df['Label_idx'], preds, target_names=[inv_label_dict[i] for i in all_indices], labels=all_indices, zero_division=0))
        evaluate_fold_composition(test_df['Label_idx'].values, preds, inv_label_dict, all_indices)

        if CLEAN_MODELS and os.path.exists(model_save_path): os.remove(model_save_path)

    # ==========================================
    # Final Exports
    # ==========================================
    if thresholds_record:
        thresh_file = os.path.join(output_dir, f"thresholds_{model_type}.json")
        with open(thresh_file, "w") as f: json.dump(thresholds_record, f, indent=4)

    print("\n" + "#"*45)
    print(f"   5-Fold CV Overall Performance ({model_type.upper()})")
    print("#"*45)

    real_labels = sorted(label_dict.values())
    final_acc = accuracy_score(overall_true, overall_pred)
    final_f1 = f1_score(overall_true, overall_pred, labels=real_labels, average='macro', zero_division=0)
    final_precision = precision_score(overall_true, overall_pred, labels=real_labels, average='macro', zero_division=0)
    final_recall = recall_score(overall_true, overall_pred, labels=real_labels, average='macro', zero_division=0)
    
    print(f"Overall Accuracy:  {final_acc:.4f}")
    print(f"Overall Precision: {final_precision:.4f} (Macro)")
    print(f"Overall Recall:    {final_recall:.4f} (Macro)")
    print(f"Overall F1-Score:  {final_f1:.4f} (Macro)\n")

    print(classification_report(overall_true, overall_pred, target_names=[inv_label_dict[i] for i in all_indices], labels=all_indices, zero_division=0))

    results_df = pd.DataFrame({'Barcodes': overall_barcodes, 'True_Label_Idx': overall_true, 'Pred_Label_Idx': overall_pred})
    results_df['True_Function'] = results_df['True_Label_Idx'].map(inv_label_dict)
    results_df['Pred_Function'] = results_df['Pred_Label_Idx'].map(inv_label_dict)
    results_df['Is_Correct'] = results_df['True_Label_Idx'] == results_df['Pred_Label_Idx']

    output_f = os.path.join(output_dir, f"cv_prediction_results_{model_type}.csv")
    results_df.to_csv(output_f, index=False)
    print(f"[*] Detailed results saved to: {output_f}")

if __name__ == "__main__":
    run_unified_cv(model_type='map', 
                   csv_file="tcrb_labels.csv", 
                   npz_file="scMACLLM_data_shuffled_compulsory.npz", 
                   output_dir="Exp_MAP_Valid_07")
