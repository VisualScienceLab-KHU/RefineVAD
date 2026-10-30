import torch
import torch.nn as nn
import torch.nn.functional as F

class TopkMILRankingLoss(nn.Module):
    def __init__(self, top_k=5, lambda1=8e-4, lambda2=8e-3):
        super(TopkMILRankingLoss, self).__init__()
        self.top_k = top_k
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, scores_abn, scores_nor):


        scores_abn = torch.sigmoid(scores_abn)
        scores_nor = torch.sigmoid(scores_nor)


        topk_abn = torch.topk(scores_abn, self.top_k, dim=1)[0].mean(dim=1)
        topk_nor = torch.topk(scores_nor, self.top_k, dim=1)[0].mean(dim=1)
        ranking_loss = F.relu(1.0 - topk_abn + topk_nor).mean()


        diff = scores_abn[:, 1:] - scores_abn[:, :-1]
        smooth_loss = torch.mean(torch.sum(diff ** 2, dim=1))


        sparse_loss = torch.mean(torch.sum(scores_abn, dim=1))


        total_loss = ranking_loss + (self.lambda1 * smooth_loss) + (self.lambda2 * sparse_loss)

        return total_loss


def gmm_losses(segment_scores, top_scores, sgmm, lambda_gmm=0.3, lambda_reg=1e-4):


    l_gmm = F.mse_loss(sgmm, segment_scores)


    l_reg = torch.mean(torch.abs(1.0 - segment_scores - top_scores))

    return lambda_gmm * l_gmm + lambda_reg * l_reg


def compute_losses(cfg, anomaly_output, normal_output, anomaly_cat_gt, normal_cat_gt):
    settings = cfg["loss"]
    anomaly_scores, anomaly_cat_logits, anomaly_top_scores, anomaly_sgmm = anomaly_output
    normal_scores, normal_cat_logits, _, _ = normal_output
    logits = torch.cat([normal_cat_logits, anomaly_cat_logits], dim=0)
    cat_label = torch.cat([normal_cat_gt, anomaly_cat_gt], dim=0)
    gt_onehot = torch.zeros((cat_label.size(0), cat_label.size(1), 2), device=logits.device)
    gt_onehot[:, :, 0] = 1.0
    anomaly_mask = cat_label.bool()
    gt_onehot[:, :, 0][anomaly_mask] = 0.0
    gt_onehot[:, :, 1][anomaly_mask] = 1.0
    weight_mask = torch.ones_like(gt_onehot)
    weight_mask[:, :, 0] = torch.where(
        gt_onehot[:, :, 0] == 0, settings["category_positive_weight"], 1.0
    )
    weight_mask[:, :, 1] = torch.where(
        gt_onehot[:, :, 1] == 1, settings["category_positive_weight"], 1.0
    )
    logit_n = logits[:, :, 0]
    logit_a = logits[:, :, 1]
    cos_sim = F.cosine_similarity(logit_n.unsqueeze(-1), logit_a.unsqueeze(-1), dim=-1)
    mutual_exclusive_loss = torch.mean(torch.clamp(cos_sim, min=0.0))
    anomaly_logits = logits[:, :, 1]
    true_mask = cat_label.bool()
    true_scores = anomaly_logits.masked_fill(~true_mask, -1e9).max(dim=1).values
    false_scores = anomaly_logits.masked_fill(true_mask, -1e9)
    top_false_scores = false_scores.max(dim=1).values
    anomaly_sample_mask = cat_label.sum(dim=1) > 0
    cat_margin_loss = F.relu(top_false_scores - true_scores + settings["category_margin"])
    cat_margin_loss = cat_margin_loss[anomaly_sample_mask].mean()
    if anomaly_sample_mask.any():
        selected_logits = anomaly_logits[anomaly_sample_mask]
        if cfg["dataset"] == "ucf_crime":
            true_indices = cat_label[anomaly_sample_mask].argmax(dim=1)
            anomaly_ce_loss = F.cross_entropy(selected_logits, true_indices)
        else:
            selected_gt = cat_label[anomaly_sample_mask].float()
            anomaly_ce_loss = F.binary_cross_entropy_with_logits(selected_logits, selected_gt)
    else:
        anomaly_ce_loss = torch.tensor(0.0, device=logits.device)
    loss_cat = (
        F.binary_cross_entropy_with_logits(logits, gt_onehot, weight=weight_mask)
        + settings["lambda_mutual"] * mutual_exclusive_loss
        + settings["lambda_category_margin"] * cat_margin_loss
        + settings["lambda_anomaly_classification"] * anomaly_ce_loss
    )
    loss_mil = TopkMILRankingLoss(
        top_k=cfg["model"]["topk"], lambda1=settings["lambda_1"], lambda2=settings["lambda_2"]
    )(anomaly_scores, normal_scores)
    loss_gmm = gmm_losses(
        segment_scores=torch.sigmoid(anomaly_scores),
        top_scores=torch.sigmoid(anomaly_top_scores),
        sgmm=torch.sigmoid(anomaly_sgmm),
        lambda_gmm=settings["lambda_gmm"], lambda_reg=settings["lambda_reg"]
    )
    total = loss_mil + settings["lambda_category"] * loss_cat
    if cfg["dataset"] == "ucf_crime":
        total = total + loss_gmm
    return {"total": total, "mil": loss_mil, "category": loss_cat, "gmm": loss_gmm}
