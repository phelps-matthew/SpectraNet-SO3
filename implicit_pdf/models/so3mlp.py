"""
MLP taking in image features and rotation query and outputting logits representing
unnormalized log of joint distribution p(R, x)
"""
import torch
from torch import nn
from torch.nn import functional as F


class SO3MLP(nn.Module):
    def __init__(self, cfg):
        super(SO3MLP, self).__init__()
        self.cfg = cfg
        self.len_img_feature = self.cfg.len_img_feature
        self.rot_dims = self.cfg.rot_dims
        self.fc_sizes = self.cfg.fc_sizes

        self.fc_img = nn.Linear(self.len_img_feature, self.fc_sizes[0])
        self.fc_rot_query = nn.Linear(self.rot_dims, self.fc_sizes[0])
        self.fc_layers = nn.ModuleList()
        self.fc_final_layer = nn.Linear(self.fc_sizes[-1], 1)

        input_size = self.fc_sizes[0]
        for size in self.fc_sizes[1:]:
            self.fc_layers.append(nn.Linear(input_size, size))
            input_size = size  # for layer (n + 1)
            self.fc_layers.append(nn.ReLU(inplace=True))


    def forward(self, img_feature, rot_query, apply_softmax=False):
        """
        Args:
            img_feature: (N, len_img_feature)
            rot_query: (N, n_queries, len_rotation)
            apply_softmax: pass as logits or softmax(logits). logits ~ log(p(R, x))

        Returns:
            (N, n_queries) as logits or softmax(logits)
        """
        x_img = self.fc_img(img_feature)
        x_rot = self.fc_rot_query(rot_query)

        # broadcast sum to combine inputs
        x = x_img[:, None, :] + x_rot
        x = F.relu(x)

        # apply mlp block to form logits
        for layer in self.fc_layers:
            x = layer(x)
        x = self.fc_final_layer(x)
        logits = x[..., 0]

        if apply_softmax:
            return F.softmax(logits, dim=-1)
        else:
            return logits 

if __name__ == "__main__":
    from torchinfo import summary
    from implicit_pdf.cfg import TrainConfig

    model = SO3MLP(TrainConfig())
    img_feature = torch.rand(32, 256)
    rot_query = torch.rand(32, 72, 9)
    logits = model(img_feature, rot_query)

    summary(model)
    print(model)
    print(logits, logits.shape)
