import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal import GlobalAdaptiveTSM


class RefineVAD(nn.Module):
    def __init__(self, args, input_dim, train=True):
        super().__init__()

        self.num_categories = args.num_categories
        self.d_emb = args.d_emb
        self.topk = args.topk
        self.fold_div = args.fold_div
        self.train_mode = train

        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, args.proj_embedding),
            nn.BatchNorm1d(args.proj_embedding),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.decision_top = nn.Linear(args.proj_embedding, 1)

        self.category_classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, args.num_categories * 2)
        )

        self.category_embedding = nn.Embedding(args.num_categories, args.d_emb)

        self.key_proj = nn.Linear(args.proj_embedding, args.d_emb)
        self.cross_attn = nn.MultiheadAttention(embed_dim=args.d_emb, num_heads=4, batch_first=True)
        self.final_proj_to_input = nn.Linear(args.d_emb, input_dim)
        self.decision_final = nn.Linear(args.proj_embedding, 1)


        if self.train_mode:
            self.GlobalAdaptiveTSM = GlobalAdaptiveTSM(fold_div=self.fold_div)
        else:
            self.GlobalAdaptiveTSM = GlobalAdaptiveTSM(input_dim=input_dim, fold_div=self.fold_div)

        self.gmm_mu = nn.Sequential(
            nn.Linear(args.proj_embedding + args.d_emb, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.gmm_sigma = nn.Sequential(
            nn.Linear(args.proj_embedding + args.d_emb, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )


    def forward(self, x, category_gt=None):
        B, N, _ = x.shape

        x = self.GlobalAdaptiveTSM(x)
        shared_feat = self.shared_backbone_forward(x)
        top_scores = self.decision_top(shared_feat).squeeze(-1)
        aggregated_feat = self.aggregate_weighted_features(x, top_scores)

        flat_logits = self.category_classifier(aggregated_feat)

        category_logits = flat_logits.view(B, -1, 2)

        cat_emb = self.get_category_embedding(category_logits, category_gt)
        proj_feat = self.cross_attention_projection(shared_feat)
        cat_context = self.cross_attention_forward(proj_feat, cat_emb)
        final_features = proj_feat + cat_context
        final_features_input_dim = self.final_proj_to_input(final_features)
        final_features_proj = self.shared_backbone_forward(final_features_input_dim)
        segment_scores = self.decision_final(final_features_proj).squeeze(-1)

        cat_expand = cat_emb.unsqueeze(1).expand(-1, final_features_proj.size(1), -1)
        gmm_input = torch.cat([final_features_proj, cat_expand], dim=-1)
        mu = self.gmm_mu(gmm_input).squeeze(-1)
        sigma = self.gmm_sigma(gmm_input).squeeze(-1)
        gmm_mask = self.generate_gaussian_mask(mu, sigma)
        sgmm = torch.diagonal(gmm_mask, dim1=1, dim2=2)

        return segment_scores, category_logits, top_scores, sgmm

    def generate_gaussian_mask(self, mu, sigma, beta=0.7):
        B, T = mu.shape
        time = torch.linspace(0, 1, T, device=mu.device).view(1, 1, T)
        mu = mu.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
        g = torch.exp(-beta * ((time - mu) ** 2) / (sigma ** 2 + 1e-6))
        return g

    def shared_backbone_forward(self, x):
        B, N, _ = x.shape
        x_flat = x.view(-1, x.shape[-1])
        feat = self.shared_backbone(x_flat)
        return feat.view(B, N, -1)


    def aggregate_weighted_features(self, x, top_scores):


        weights = F.softmax(top_scores, dim=1).unsqueeze(-1)

        emphasized = x * (1.0 + weights)
        aggregated = emphasized.mean(dim=1)

        return aggregated


    def get_category_embedding(self, category_logits, category_gt=None):


        cat_probs = F.softmax(category_logits, dim=2)
        anomaly_probs = cat_probs[:, :, 1]


        max_anom, _ = anomaly_probs.max(dim=1)
        is_normal = max_anom < 0.5


        uniform = anomaly_probs.new_full(anomaly_probs.size(), 1.0 / self.num_categories)
        weights = anomaly_probs.clone()
        weights[is_normal] = uniform[is_normal]


        emb_weights = self.category_embedding.weight[:self.num_categories]
        weighted_emb = weights.matmul(emb_weights)


        if self.training and category_gt is not None:
            gt_idx = torch.argmax(category_gt, dim=1)
            gt_emb = self.category_embedding(gt_idx)
            return gt_emb + weighted_emb
        else:
            return weighted_emb

    def cross_attention_projection(self, shared_feat):
        return self.key_proj(shared_feat)

    def cross_attention_forward(self, proj_feat, cat_emb):
        query = cat_emb.unsqueeze(1)
        attn_out, _ = self.cross_attn(query, proj_feat, proj_feat)
        return attn_out.expand(-1, proj_feat.size(1), -1)
