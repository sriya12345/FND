import torch
import torch.nn as nn
import clip

class CLIPFakeNewsClassifier(nn.Module):
    def __init__(self):                                        # load/define the models 
        super().__init__()
        # load CLIP here
        self.clip_model, _ = clip.load("ViT-B/32")
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # define MLP here
        self.mlp = nn.Sequential(
            nn.Linear(1025,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,1)
        )
        

    def forward(self, image, text):                           # perform computation (encode -> concatenate -> classify)
        text_tokens = clip.tokenize(text, truncate = True).to(next(self.parameters()).device)     
        image_emb = self.clip_model.encode_image(image).float()
        text_emb = self.clip_model.encode_text(text_tokens).float()
        similarity = torch.nn.functional.cosine_similarity(image_emb, text_emb)
        combined = torch.cat([image_emb,text_emb,similarity.unsqueeze(1)], dim=1)   # concatenation
        output = self.mlp(combined)
        return output