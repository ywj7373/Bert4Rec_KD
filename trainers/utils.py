import torch


def recall(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / labels.sum(1).float()).mean().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(n, k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean().item()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    scores = scores.cpu()
    labels = labels.cpu()
    answer_count = labels.sum(1)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights).sum(1)
        idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count])
        ndcg = (dcg / idcg).mean().item()
        metrics['NDCG@%d' % k] = ndcg

    return metrics


def extract_layers(layers, state_dict):
    distill_sd = {}

    distill_sd["bert.embedding.token.weight"] = state_dict["bert.embedding.token.weight"]
    distill_sd["bert.embedding.position.pe.weight"] = state_dict["bert.embedding.position.pe.weight"]

    for layer in layers:
        for idx in range(3):
            for wb in ["weight", "bias"]:
                distill_sd[f"bert.transformer_blocks.{layer}.attention.linear_layers.{idx}.{wb}"] = state_dict[
                    f"bert.transformer_blocks.{layer}.attention.linear_layers.{idx}.{wb}"]

        for wb in ["weight", "bias"]:
            distill_sd[f"bert.transformer_blocks.{layer}.attention.output_linear.{wb}"] = state_dict[
                f"bert.transformer_blocks.{layer}.attention.output_linear.{wb}"]
            distill_sd[f"bert.transformer_blocks.{layer}.feed_forward.w_1.{wb}"] = state_dict[
                f"bert.transformer_blocks.{layer}.feed_forward.w_1.{wb}"]
            distill_sd[f"bert.transformer_blocks.{layer}.feed_forward.w_2.{wb}"] = state_dict[
                f"bert.transformer_blocks.{layer}.feed_forward.w_2.{wb}"]

        for ab in ["a_2", "b_2"]:
            distill_sd[f"bert.transformer_blocks.{layer}.input_sublayer.norm.{ab}"] = state_dict[
                f"bert.transformer_blocks.{layer}.input_sublayer.norm.{ab}"]
            distill_sd[f"bert.transformer_blocks.{layer}.output_sublayer.norm.{ab}"] = state_dict[
                f"bert.transformer_blocks.{layer}.output_sublayer.norm.{ab}"]

    distill_sd["out.weight"] = state_dict["out.weight"]
    distill_sd["out.bias"] = state_dict["out.bias"]

    return distill_sd
