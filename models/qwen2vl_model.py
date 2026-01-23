import torch 
import torch.nn.functional as F
from .qwen2_vl.qwen2_vl_wrapper import Qwen2VLForEmbedding, Qwen2VLProcessorForEmbedding

class Qwen2VLModel():
    def __init__(self, model_name_or_path, device='cuda', **kwargs):
        self.model = Qwen2VLForEmbedding.from_pretrained(
            model_name_or_path, 
            device_map=device, 
            torch_dtype=torch.bfloat16
        )
        self.model.eval()
        self.processor = Qwen2VLProcessorForEmbedding.from_pretrained(
            model_name_or_path, 
            instruction_standalone=True,
            max_length=3024,
            min_pixels=32*32*4,
            max_pixels=32*32*1280,
            total_pixels=32*32*4500,
            num_frames=48
        )
    @torch.no_grad()
    def query_emb(self, texts, images, instruction="",batch_size=32):
        assert len(texts) == len(images)
        query_embs = []
        for i in range(0, len(texts), batch_size):
            tmp_texts = texts[i:i+batch_size]
            tmp_images = images[i:i+batch_size]
            inputs = self.processor(texts=tmp_texts,  images=tmp_images, videos=[None]*len(tmp_texts), instruction=[instruction]*len(tmp_texts), is_query=False, max_length=1024, truncation=False)
            inputs = inputs.to(self.model.device)
            embs = self.model(**inputs)
            query_embs.append(embs)
        
        query_embs = torch.cat(query_embs, dim=0)
        query_embs = F.normalize(query_embs, p=2, dim=-1)

        return query_embs
        
    @torch.no_grad()    
    def corpus_emb(self, images, instruction="", batch_size=32):
        corpus_embs = []
        for i in range(0, len(images), batch_size):
            tmp_images = images[i:i+batch_size]
            inputs = self.processor(texts=[None]*len(tmp_images),  images=tmp_images, videos=[None]*len(tmp_images), instruction=[instruction]*len(tmp_images), is_query=False, max_length=1024, truncation=False)
            inputs = inputs.to(self.model.device)
            embs = self.model(**inputs)
            corpus_embs.append(embs)
        
        corpus_embs = torch.cat(corpus_embs, dim=0)
        corpus_embs = F.normalize(corpus_embs, p=2, dim=-1)

        return corpus_embs