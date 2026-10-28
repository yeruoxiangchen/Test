import torch.nn.functional as F


def compatible_extractor(model_path):
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
    class Extractor(DinoV3FeatureExtractor):
        def extract_features(self,image):
            image=image.to(self.model.embeddings.patch_embeddings.weight.dtype)
            hidden=self.model.embeddings(image,bool_masked_pos=None)
            position=self.model.rope_embeddings(image)
            if hasattr(self.model,'layer'):
                for layer in self.model.layer:
                    hidden=layer(hidden,position_embeddings=position)
            else:
                # Transformers5 moved the blocks under an encoder. Use its
                # PRE-final-norm output, retaining upstream's parameter-free
                # layer_norm rather than accidentally applying learned norm.
                hidden=self.model.model(hidden,position_embeddings=position).last_hidden_state
            return F.layer_norm(hidden,hidden.shape[-1:])
    return Extractor(str(model_path),image_size=512)
