"""DINOv3 layer-container compatibility; preserve upstream prenorm + functional LN."""
import torch.nn.functional as F
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

class NativeDinoV3FeatureExtractor(DinoV3FeatureExtractor):
    def extract_features(self,image):
        image=image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        h=self.model.embeddings(image,bool_masked_pos=None)
        positions=self.model.rope_embeddings(image)
        layers=getattr(self.model,'layer',None)
        if layers is None:layers=self.model.model.layer
        for layer in layers:
            h=layer(h,position_embeddings=positions)
        return F.layer_norm(h,h.shape[-1:])
