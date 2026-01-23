
from .rzen.rzen_embed_inference import RzenEmbed
import torch
import torch.nn.functional as F

class RzenModel():
    def __init__(self, model_name_or_path, device='cuda', **kwargs):
        self.model = RzenEmbed(model_name_or_path).to(device)

    def query_emb(self, texts, images, instruction = "Given an image, find a similar everyday image with the described changes: ", batch_size=32):
        assert len(texts) == len(images)
        query_embs = []
        for i in range(0, len(texts), batch_size):
            embs = self.model.get_fused_embeddings(instruction=instruction, texts=texts[i:i+batch_size], images=images[i:i+batch_size])
            query_embs.append(embs)
        
        query_embs = torch.cat(query_embs, dim=0)
        query_embs = F.normalize(query_embs, dim=-1)
        return query_embs

    def corpus_emb(self, images, instruction="Represent the given image. ", batch_size=32):
        corpus_embs = []
        for i in range(0, len(images), batch_size):
            embs = self.model.get_image_embeddings(instruction=instruction, images=images[i:i+batch_size])
            corpus_embs.append(embs)

        corpus_embs = torch.cat(corpus_embs, dim=0)
        corpus_embs = F.normalize(corpus_embs, dim=-1)
        return corpus_embs 