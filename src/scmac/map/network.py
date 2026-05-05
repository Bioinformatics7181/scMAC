# -------------------------------------------------------------------------
# Name: network.py
# Coding: utf8
# Author: Xinyang Qian
# Intro: Containing deep learning network classes.
# -------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import math
import os
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import utils


class TCRD(nn.Module):
    # TCR detector (TCRD) can predict a TCR sequence's class (e.g. cancer-associated TCR) (binary classification).
    # It can extract the antigen-specific biochemical features of TCRs based on the convolutional neural network.
    def __init__(self, aa_num=24, feature_num=15, kernel_size=None, filter_num=None, filters_num=1, class_num=2, drop_out=0.4):
        super(TCRD, self).__init__()
        self.aa_num = aa_num  # The number of amino acids that one TCR contains.
        self.feature_num = feature_num  # The dimension of the feature vector of one amino acid.
        if kernel_size is None:
            kernel_size = [2, 3, 4, 5, 6, 7]
        self.kernel_size = kernel_size  # The specification of the convolution kernel in the convolution layer.
        if filter_num is None:
            filter_num = [3, 3, 3, 2, 2, 1]
        assert len(filter_num) == len(kernel_size), \
            "The parameters 'kernel_size' and 'filter_num' set do not match!"
        self.filter_num = []  # The number of the corresponding convolution kernels.
        for ftr in filter_num:
            self.filter_num.append(ftr * filters_num)
        self.class_num = class_num
        self.drop_out = drop_out
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(self.kernel_size)
        ])
        self.fc = nn.Linear(sum(self.filter_num), self.class_num)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, sum(self.filter_num))
        out = self.dropout(self.fc(out))
        return out

    @staticmethod
    def training(tcrs_train, lbs_train, tcrs_test, lbs_test,
                 filters_n, class_num, lr, ep, dropout, log_inr, model_f,
                 aa_f, device, shuffle=True, is_training_mode=False, batch_size=512):
        # 1. Prepare Training DataLoader
        training_sps = [[[tcr]] for tcr in tcrs_train]
        aa_v = utils.get_features(aa_f)  
        input_batch, label_batch = utils.generate_input_for_training(training_sps, lbs_train, aa_v, ins_num=1)
        
        input_tensor = torch.Tensor(input_batch)
        label_tensor = torch.LongTensor(label_batch)
        train_dataset = Data.TensorDataset(input_tensor, label_tensor)
        train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        
        model = TCRD(filters_num=filters_n, drop_out=dropout, class_num=class_num).to(device)
        criterion = nn.CrossEntropyLoss().to(device)  
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # --- Internal helper for online evaluation ---
        def _evaluate_local(eval_tcrs, eval_lbs):
            model.eval()
            eval_sps = [[[tcr]] for tcr in eval_tcrs]
            eval_input, _ = utils.generate_input_for_training(eval_sps, eval_lbs, aa_v, ins_num=1)
            eval_loader = Data.DataLoader(Data.TensorDataset(torch.Tensor(eval_input)), batch_size=batch_size, shuffle=False)
            
            all_preds = []
            with torch.no_grad():
                for batch_x in eval_loader:
                    batch_x = batch_x[0].to(device)
                    out = model(batch_x)
                    preds = torch.argmax(out, dim=1).cpu().numpy().tolist()
                    all_preds.extend(preds)
            model.train()
            
            acc = accuracy_score(eval_lbs, all_preds)
            f1 = f1_score(eval_lbs, all_preds, average='macro', zero_division=0)
            return acc, f1

        best_test_f1 = -1.0
        best_train_f1 = -1.0

        model.train()
        for epoch in range(ep):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # --- Online Evaluation Reporting ---
            if (epoch + 1) % log_inr == 0:
                avg_loss = epoch_loss / len(train_loader)
                
                # Internal evaluation
                train_acc, train_f1 = _evaluate_local(tcrs_train, lbs_train)
                test_acc, test_f1 = _evaluate_local(tcrs_test, lbs_test)
                
                print(f"Epoch: {epoch + 1:04d} | Loss: {avg_loss:.4f} | "
                      f"Train [Acc: {train_acc:.4f}, F1: {train_f1:.4f}] | "
                      f"Test [Acc: {test_acc:.4f}, F1: {test_f1:.4f}]")

                if is_training_mode:
                    if test_f1 > best_test_f1:
                        best_test_f1 = test_f1
                        torch.save(model.state_dict(), model_f)
                else:
                    if train_f1 > best_train_f1:
                        best_train_f1 = train_f1
                        torch.save(model.state_dict(), model_f)
                        
        if not is_training_mode:
            return best_train_f1
        else:
            return best_test_f1

    @staticmethod
    def prediction(tcrs, filters_n, class_num, model_f, aa_f, device):
        model = TCRD(filters_num=filters_n, class_num=class_num).to(device)
        model.load_state_dict(torch.load(model_f))
        model = model.eval()
        
        tcr_scores = []
        aa_v = utils.get_features(aa_f)  
        for tcr in tcrs:
            input_x = utils.generate_input_for_prediction([[tcr]], aa_v, ins_num=1)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            
            with torch.no_grad():
                predict = model(input_x)
                prob = F.softmax(predict, dim=1)[0].cpu().numpy().tolist() 
            tcr_scores.append(prob)
        return tcr_scores


class scMACLoss(nn.Module):
    def __init__(self, alpha_weights=None, gamma=2.0, w_align=1.0, w_proto=0.5, w_kld=0.01):
        super(scMACLoss, self).__init__()
        self.alpha = alpha_weights
        self.gamma = gamma
        self.w_align = w_align
        self.w_proto = w_proto
        self.w_kld = w_kld

    def forward(self, outputs, z_gold, labels, prototypes):
        z_tcr = outputs['z_tcr']
        logits = outputs['logits']
        mu = outputs['mu']
        logvar = outputs['logvar']

        # 1. Focal Loss (Ignores label 0 if its weight is 0.0)
        ce_loss = F.cross_entropy(logits, labels, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        loss_focal = ((1 - pt) ** self.gamma * ce_loss).mean()

        # 2. Latent Alignment
        loss_align = F.mse_loss(z_tcr, z_gold) + (1 - F.cosine_similarity(z_tcr, z_gold).mean())

        # 3. Prototype Constraint
        batch_proto = prototypes[labels]
        mask = (labels != 0).float().unsqueeze(1)
        loss_proto = (F.mse_loss(z_tcr, batch_proto, reduction='none') * mask).mean()

        # 4. KL Divergence
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_focal + self.w_align * loss_align + self.w_proto * loss_proto + self.w_kld * kld


class scMACMap(nn.Module):
    def __init__(self, v_vocab_size=1, j_vocab_size=1, latent_dim=128, class_num=7, 
                 aa_num=24, feature_num=15, kernel_size=None, filter_num=None, 
                 filters_num=1):
        super(scMACMap, self).__init__()
        
        self.aa_num = aa_num
        self.feature_num = feature_num
        
        if kernel_size is None:
            kernel_size = [2, 3, 4, 5, 6, 7]
        self.kernel_size = kernel_size
        
        if filter_num is None:
            filter_num = [3, 3, 3, 2, 2, 1]
            
        self.filter_num = []
        for ftr in filter_num:
            self.filter_num.append(ftr * filters_num)
            
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.feature_num,
                          out_channels=self.filter_num[idx],
                          kernel_size=h,
                          stride=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
            for idx, h in enumerate(self.kernel_size)
        ])
        
        cnn_out_dim = sum(self.filter_num)
        
        # padding_idx=0 ensures unseen genes are zeroed out
        self.v_embed = nn.Embedding(v_vocab_size, 48, padding_idx=0)
        self.j_embed = nn.Embedding(j_vocab_size, 16, padding_idx=0)
        
        fusion_dim = cnn_out_dim + 48 + 16
            
        self.fc_hidden = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, class_num)
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, cdr3_x, v_idx, j_idx):
        cdr3_x = cdr3_x.reshape(-1, self.feature_num, self.aa_num)
        
        cnn_outputs = [conv(cdr3_x) for conv in self.convs]
        cnn_out = torch.cat(cnn_outputs, dim=1)
        cnn_out = cnn_out.reshape(-1, sum(self.filter_num))
        
        v_feat = self.v_embed(v_idx)
        j_feat = self.j_embed(j_idx)
            
        f_combined = torch.cat([cnn_out, v_feat, j_feat], dim=1)
        
        h = self.fc_hidden(f_combined)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z_tcr = self.reparameterize(mu, logvar)
        
        logits = self.classifier(z_tcr)
        
        return {"z_tcr": z_tcr, "mu": mu, "logvar": logvar, "logits": logits}

    @staticmethod
    def training(train_loader, test_loader, train_eval_loader, y_train, y_test,
                 prototypes, class_weights, v_vocab_size, j_vocab_size, 
                 latent_dim, filters_n, class_num, lr, ep, model_f, device,
                 w_align=1.0, w_proto=0.5, w_kld=0.01, log_inr=10, is_training_mode=False):     
        t_prototypes = torch.as_tensor(prototypes, dtype=torch.float32, device=device)
        t_class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        
        model = scMACMap(v_vocab_size=v_vocab_size, j_vocab_size=j_vocab_size, latent_dim=latent_dim,
                         class_num=class_num, filters_num=filters_n).to(device)
        
        criterion = scMACLoss(alpha_weights=t_class_weights, 
                              w_align=w_align, 
                              w_proto=w_proto, 
                              w_kld=w_kld).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # --- Internal helper for online evaluation ---
        def _evaluate(eval_model, dataloader):
            eval_model.eval()
            all_preds = []
            with torch.no_grad():
                # scMACMap DataLoader yields 5 items
                for b_cdr3, b_v, b_j, _, _ in dataloader:
                    b_cdr3, b_v, b_j = b_cdr3.to(device), b_v.to(device), b_j.to(device)
                    outputs = eval_model(b_cdr3, b_v, b_j)
                    # Argmax over known classes (index 1 to N-1), then shift index back (+1)
                    preds = torch.argmax(outputs['logits'][:, 1:], dim=1) + 1
                    all_preds.extend(preds.cpu().numpy().tolist())
            eval_model.train() # Revert to train mode
            return all_preds

        best_test_f1 = -1.0
        best_train_f1 = -1.0

        model.train()
        for epoch in range(ep):
            epoch_loss = 0.0
            for b_cdr3, b_v, b_j, b_zgold, b_labels in train_loader:
                b_cdr3 = b_cdr3.to(device)
                b_v = b_v.to(device)
                b_j = b_j.to(device)
                b_zgold = b_zgold.to(device)
                b_labels = b_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(b_cdr3, b_v, b_j)
                loss = criterion(outputs, b_zgold, b_labels, t_prototypes)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            # --- Online Evaluation Reporting ---
            if (epoch + 1) % log_inr == 0:
                avg_loss = epoch_loss / len(train_loader)
                
                # Predict on Train and Test sets
                train_preds = _evaluate(model, train_eval_loader)
                test_preds = _evaluate(model, test_loader)
                
                # Calculate metrics
                train_acc = accuracy_score(y_train, train_preds)
                train_f1 = f1_score(y_train, train_preds, average='macro', zero_division=0)
                
                test_acc = accuracy_score(y_test, test_preds)
                test_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)
                
                print(f"Epoch: {epoch + 1:04d} | Loss: {avg_loss:.4f} | "
                      f"Train [Acc: {train_acc:.4f}, F1: {train_f1:.4f}] | "
                      f"Test [Acc: {test_acc:.4f}, F1: {test_f1:.4f}]")

                if is_training_mode:
                    if test_f1 > best_test_f1:
                        best_test_f1 = test_f1
                        torch.save(model.state_dict(), model_f)
                else:
                    if train_f1 > best_train_f1:
                        best_train_f1 = train_f1
                        torch.save(model.state_dict(), model_f)
                        
        if not is_training_mode:
            return best_train_f1
        else:
            return best_test_f1

    @staticmethod
    def prediction(test_loader, v_vocab_size, j_vocab_size, latent_dim, filters_n, class_num, 
                   model_f, device):
        
        model = scMACMap(v_vocab_size=v_vocab_size, j_vocab_size=j_vocab_size, latent_dim=latent_dim, 
                         class_num=class_num, filters_num=filters_n).to(device)
        model.load_state_dict(torch.load(model_f))
        model.eval()
        
        all_probs = []
        all_uncertainties = []
        
        with torch.no_grad():
            for b_cdr3, b_v, b_j, _, _ in test_loader:
                b_cdr3 = b_cdr3.to(device)
                b_v = b_v.to(device)
                b_j = b_j.to(device)
                
                outputs = model(b_cdr3, b_v, b_j)
                
                probs = F.softmax(outputs['logits'], dim=1).cpu().numpy().tolist()
                uncertainties = torch.exp(outputs['logvar']).mean(dim=1).cpu().numpy().tolist()
                
                all_probs.extend(probs)
                all_uncertainties.extend(uncertainties)
                
        return all_probs, all_uncertainties


class CGCLLoss(nn.Module):
    """
    Clonal-Aware Cross-Modal Contrastive Loss.
    Aligns TCR embeddings with LLM embeddings using a clonal mask to handle 1-to-many ambiguity.
    """
    def __init__(self, alpha_weights=None, gamma=2.0, w_contrast=1.0, w_kld=0.01, temperature=0.07):
        super(CGCLLoss, self).__init__()
        self.alpha = alpha_weights
        self.gamma = gamma
        self.w_contrast = w_contrast
        self.w_kld = w_kld
        self.temperature = temperature

    def forward(self, outputs, z_gold, labels, tcr_ids):
        z_projected = outputs['z_projected']
        logits = outputs['logits']
        mu = outputs['mu']
        logvar = outputs['logvar']

        # 1. Focal Loss for classification (Maintains evaluation baseline)
        ce_loss = F.cross_entropy(logits, labels, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        loss_focal = ((1 - pt) ** self.gamma * ce_loss).mean()

        # 2. Clonal-Aware Contrastive Loss (InfoNCE variant)
        # Normalize representations to unit sphere
        z_proj_norm = F.normalize(z_projected, dim=-1)
        z_gold_norm = F.normalize(z_gold, dim=-1)
        
        # Compute similarity matrix (Batch_size x Batch_size)
        logits_sim = torch.matmul(z_proj_norm, z_gold_norm.T) / self.temperature
        
        # Build Clonal Mask: mask[i, j] = 1 if sample i and j share the same TCR sequence
        mask = torch.eq(tcr_ids.unsqueeze(1), tcr_ids.unsqueeze(0)).float()
        
        # Calculate log probabilities
        log_prob = logits_sim - torch.log(torch.exp(logits_sim).sum(dim=1, keepdim=True) + 1e-8)
        
        # Mean log-likelihood for positive pairs (handling 1-to-many mappings)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        loss_contrast = -mean_log_prob_pos.mean()

        # 3. KL Divergence for VAE regularization
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_focal + self.w_contrast * loss_contrast + self.w_kld * kld


class scMAC_CGCL(nn.Module):
    """
    scMAC architecture with a Non-linear Projector for Contrastive Learning.
    """
    def __init__(self, v_vocab_size=1, j_vocab_size=1, latent_dim=128, class_num=7, 
                 aa_num=24, feature_num=15, kernel_size=None, filter_num=None, 
                 filters_num=1):
        super(scMAC_CGCL, self).__init__()
        
        self.aa_num = aa_num
        self.feature_num = feature_num
        
        if kernel_size is None:
            kernel_size = [2, 3, 4, 5, 6, 7]
        self.kernel_size = kernel_size
        
        if filter_num is None:
            filter_num = [3, 3, 3, 2, 2, 1]
            
        self.filter_num = [ftr * filters_num for ftr in filter_num]
            
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.feature_num,
                          out_channels=self.filter_num[idx],
                          kernel_size=h,
                          stride=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
            for idx, h in enumerate(self.kernel_size)
        ])
        
        cnn_out_dim = sum(self.filter_num)
        
        self.v_embed = nn.Embedding(v_vocab_size, 48, padding_idx=0)
        self.j_embed = nn.Embedding(j_vocab_size, 16, padding_idx=0)
        
        fusion_dim = cnn_out_dim + 48 + 16
            
        self.fc_hidden = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # --- NEW: Non-linear Projector ---
        # Maps the VAE latent space to the LLM semantic space (e.g., 64-dim)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, latent_dim)
        )
        
        # Classifier for benchmarking Macro-F1
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, class_num)
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, cdr3_x, v_idx, j_idx):
        cdr3_x = cdr3_x.reshape(-1, self.feature_num, self.aa_num)
        
        cnn_outputs = [conv(cdr3_x) for conv in self.convs]
        cnn_out = torch.cat(cnn_outputs, dim=1)
        cnn_out = cnn_out.reshape(-1, sum(self.filter_num))
        
        v_feat = self.v_embed(v_idx)
        j_feat = self.j_embed(j_idx)
            
        f_combined = torch.cat([cnn_out, v_feat, j_feat], dim=1)
        
        h = self.fc_hidden(f_combined)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z_tcr = self.reparameterize(mu, logvar)
        
        z_projected = self.projector(z_tcr)
        logits = self.classifier(z_tcr)
        
        return {"z_tcr": z_tcr, "z_projected": z_projected, "mu": mu, "logvar": logvar, "logits": logits}

    # @staticmethod
    # def training(train_loader, class_weights, v_vocab_size, j_vocab_size, 
    #              latent_dim, filters_n, class_num, lr, ep, model_f, device,
    #              w_contrast=1.0, w_kld=0.01, temp=0.07):
        
    #     t_class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    #     model = scMAC_CGCL(v_vocab_size=v_vocab_size, j_vocab_size=j_vocab_size, latent_dim=latent_dim,
    #                        class_num=class_num, filters_num=filters_n).to(device)
        
    #     criterion = CGCLLoss(alpha_weights=t_class_weights, w_contrast=w_contrast, 
    #                          w_kld=w_kld, temperature=temp).to(device)
    #     optimizer = optim.Adam(model.parameters(), lr=lr)
        
    #     model.train()
    #     for epoch in range(ep):
    #         epoch_loss = 0.0
    #         for b_cdr3, b_v, b_j, b_zgold, b_labels, b_tcrid in train_loader:
    #             b_cdr3, b_v, b_j = b_cdr3.to(device), b_v.to(device), b_j.to(device)
    #             b_zgold, b_labels = b_zgold.to(device), b_labels.to(device)
    #             b_tcrid = b_tcrid.to(device)
                
    #             optimizer.zero_grad()
    #             outputs = model(b_cdr3, b_v, b_j)
    #             loss = criterion(outputs, b_zgold, b_labels, b_tcrid)
                
    #             loss.backward()
    #             optimizer.step()
    #             epoch_loss += loss.item()
                
    #         if (epoch + 1) % 50 == 0:
    #             print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(epoch_loss / len(train_loader)))
                
    #     torch.save(model.state_dict(), model_f)
    #     return 0


    # Debug
    @staticmethod
    def training(train_loader, test_loader, train_eval_loader, y_train, y_test,
                 class_weights, v_vocab_size, j_vocab_size, 
                 latent_dim, filters_n, class_num, lr, ep, model_f, device,
                 w_contrast=1.0, w_kld=0.01, temp=0.07, log_inr=10, is_training_mode=False):      
        t_class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        model = scMAC_CGCL(v_vocab_size=v_vocab_size, j_vocab_size=j_vocab_size, latent_dim=latent_dim,
                           class_num=class_num, filters_num=filters_n).to(device)
        
        criterion = CGCLLoss(alpha_weights=t_class_weights, w_contrast=w_contrast, 
                             w_kld=w_kld, temperature=temp).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # --- Internal helper for online evaluation ---
        def _evaluate(eval_model, dataloader):
            eval_model.eval()
            all_preds = []
            with torch.no_grad():
                for b_cdr3, b_v, b_j, _, _, _ in dataloader:
                    b_cdr3, b_v, b_j = b_cdr3.to(device), b_v.to(device), b_j.to(device)
                    outputs = eval_model(b_cdr3, b_v, b_j)
                    # Argmax over known classes (index 1 to N-1), then shift index back (+1)
                    preds = torch.argmax(outputs['logits'][:, 1:], dim=1) + 1
                    all_preds.extend(preds.cpu().numpy().tolist())
            eval_model.train() # Revert to train mode
            return all_preds

        best_test_f1 = -1.0
        best_train_f1 = -1.0

        model.train()
        for epoch in range(ep):
            epoch_loss = 0.0
            for b_cdr3, b_v, b_j, b_zgold, b_labels, b_tcrid in train_loader:
                b_cdr3, b_v, b_j = b_cdr3.to(device), b_v.to(device), b_j.to(device)
                b_zgold, b_labels = b_zgold.to(device), b_labels.to(device)
                b_tcrid = b_tcrid.to(device)
                
                optimizer.zero_grad()
                outputs = model(b_cdr3, b_v, b_j)
                loss = criterion(outputs, b_zgold, b_labels, b_tcrid)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            # --- Online Evaluation Reporting ---
            if (epoch + 1) % log_inr == 0:
                avg_loss = epoch_loss / len(train_loader)
                
                # Predict on Train and Test sets
                train_preds = _evaluate(model, train_eval_loader)
                test_preds = _evaluate(model, test_loader)
                
                # Calculate metrics
                train_acc = accuracy_score(y_train, train_preds)
                train_f1 = f1_score(y_train, train_preds, average='macro', zero_division=0)
                
                test_acc = accuracy_score(y_test, test_preds)
                test_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)
                
                print(f"Epoch: {epoch + 1:04d} | Loss: {avg_loss:.4f} | "
                      f"Train [Acc: {train_acc:.4f}, F1: {train_f1:.4f}] | "
                      f"Test [Acc: {test_acc:.4f}, F1: {test_f1:.4f}]")

                if is_training_mode:
                    if test_f1 > best_test_f1:
                        best_test_f1 = test_f1
                        torch.save(model.state_dict(), model_f)
                else:
                    if train_f1 > best_train_f1:
                        best_train_f1 = train_f1
                        torch.save(model.state_dict(), model_f)
                
        if not is_training_mode:
            # Test mode behavior: Return the historically best training F1 score
            return best_train_f1
        else:
            # Training mode behavior: Return the historically best test F1 score
            return best_test_f1
    # Debug


    @staticmethod
    def prediction(test_loader, v_vocab_size, j_vocab_size, latent_dim, filters_n, class_num, 
                   model_f, device):
        model = scMAC_CGCL(v_vocab_size=v_vocab_size, j_vocab_size=j_vocab_size, latent_dim=latent_dim, 
                           class_num=class_num, filters_num=filters_n).to(device)
        model.load_state_dict(torch.load(model_f))
        model.eval()
        
        all_probs, all_uncertainties = [], []
        
        with torch.no_grad():
            for b_cdr3, b_v, b_j, _, _, _ in test_loader:
                b_cdr3, b_v, b_j = b_cdr3.to(device), b_v.to(device), b_j.to(device)
                outputs = model(b_cdr3, b_v, b_j)
                
                probs = F.softmax(outputs['logits'], dim=1).cpu().numpy().tolist()
                uncertainties = torch.exp(outputs['logvar']).mean(dim=1).cpu().numpy().tolist()
                
                all_probs.extend(probs)
                all_uncertainties.extend(uncertainties)
                
        return all_probs, all_uncertainties


from loss import mmd_rbf

class MIST_TCRb_LLM(nn.Module):
    def __init__(self, llm_dim, class_num, aa_num=30, feature_num=15, filters_num=1, latent_dim=128):
        super(MIST_TCRb_LLM, self).__init__()
        self.latent_dim = latent_dim
        
        # 1. TCR encoder based on the existing TCRD backbone.
        # Set class_num to latent_dim so the module outputs a latent representation.
        self.tcr_encoder = TCRD(
            aa_num=aa_num, 
            feature_num=feature_num, 
            filters_num=filters_num, 
            class_num=latent_dim, 
            drop_out=0.2
        )
        
        # 2. LLM aligner projects the scMAC-LLM reference into the same latent space.
        self.llm_aligner = nn.Sequential(
            nn.Linear(llm_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # 3. Functional classification head for 7-state prediction.
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, class_num)
        )

    def forward(self, cdr3b, v_idx, j_idx, llm_latent=None, mode='joint'):
        # Extract TCR features.
        # V/J indices are ignored here to keep the baseline TCRD-style encoder unchanged.
        z_tcr = self.tcr_encoder(cdr3b)
        
        if mode == 'joint' and llm_latent is not None:
            # Map the reference latent target.
            z_llm = self.llm_aligner(llm_latent)
            
            # Functional prediction.
            cls_tcr = self.classifier(z_tcr)
            cls_llm = self.classifier(z_llm)
            
            return z_tcr, z_llm, cls_tcr, cls_llm
        else:
            # Inference mode returns a dictionary compatible with mapping_unified.
            cls_tcr = self.classifier(z_tcr)
            return {'latent': z_tcr, 'logits': cls_tcr, 'uncertainty': torch.zeros(z_tcr.size(0))}

    @staticmethod
    def train_network(train_loader, test_loader, v_vocab_size, j_vocab_size, llm_dim, class_num, 
                      epochs, lr, is_training_mode, model_f, device, w_mmd=1.0, w_kl=0.0):
        # Initialize model.
        model = MIST_TCRb_LLM(llm_dim=llm_dim, class_num=class_num, aa_num=30).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        best_test_f1 = -1.0
        best_train_f1 = -1.0
        
        for epoch in range(epochs):
            model.train()
            for b_cdr3, _, _, b_llm, b_labels, _ in train_loader:
                b_cdr3, b_llm, b_labels = b_cdr3.to(device), b_llm.to(device), b_labels.to(device)
                
                optimizer.zero_grad()
                # Use joint mode for distillation-style training.
                z_tcr, z_llm, cls_tcr, cls_llm = model(b_cdr3, None, None, llm_latent=b_llm, mode='joint')
                
                # Main loss: MMD alignment plus classification loss.
                loss_mmd = mmd_rbf(z_tcr, z_llm)
                loss_cls = F.cross_entropy(cls_tcr, b_labels) + F.cross_entropy(cls_llm, b_labels)
                
                total_loss = w_mmd * loss_mmd + loss_cls
                total_loss.backward()
                optimizer.step()

            # Evaluation follows the same logic as the mapping module.
            model.eval()
            with torch.no_grad():
                # Compute test F1.
                test_preds, test_trues = [], []
                for b_cdr3, _, _, _, b_labels, _ in test_loader:
                    out = model(b_cdr3.to(device), None, None, mode='tcr_only')
                    test_preds.extend(torch.argmax(out['logits'], dim=1).cpu().numpy())
                    test_trues.extend(b_labels.numpy())
                t_f1 = f1_score(test_trues, test_preds, average='macro', zero_division=0)
                
                # Simplified logging: only test evaluation is shown here.
                if is_training_mode:
                    if t_f1 > best_test_f1:
                        best_test_f1 = t_f1
                        torch.save(model.state_dict(), model_f)
                else:
                    # In non-training mode, save according to the default checkpoint logic.
                    torch.save(model.state_dict(), model_f)
                    best_test_f1 = t_f1

        return best_test_f1

    @staticmethod
    def prediction(test_loader, v_vocab_size, j_vocab_size, llm_dim, class_num, model_f, device):
        # Prediction is compatible with mapping_unified.
        model = MIST_TCRb_LLM(llm_dim=llm_dim, class_num=class_num).to(device)
        model.load_state_dict(torch.load(model_f))
        model.eval()
        
        all_probs, all_uncertainties = [], []
        with torch.no_grad():
            for b_cdr3, _, _, _, _, _ in test_loader:
                out = model(b_cdr3.to(device), None, None, mode='tcr_only')
                probs = F.softmax(out['logits'], dim=1).cpu().numpy().tolist()
                all_probs.extend(probs)
                all_uncertainties.extend([0.0] * len(probs))  # TCRD is deterministic.
        return all_probs, all_uncertainties
