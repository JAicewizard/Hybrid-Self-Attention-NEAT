import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

np.set_printoptions(linewidth=100)


class SelfAttentionModule(nn.Module):
    def __init__(self, input_d, transformer_d):
        super(SelfAttentionModule, self).__init__()
        self.__layers = []

        self.__fc_q = nn.Linear(input_d, transformer_d)
        self.__layers.append(self.__fc_q)
        self.__fc_k = nn.Linear(input_d, transformer_d)
        self.__layers.append(self.__fc_k)

    def forward(self, input_data):
        _, _, input_d = input_data.size()

        # Linear transforms
        queries = self.__fc_q(input=input_data)
        keys = self.__fc_k(input=input_data)

        # Attention matrix
        dot = torch.bmm(queries, keys.transpose(1, 2))
        scaled_dot = torch.div(dot, torch.sqrt(torch.tensor(input_d).float()))

        attention_weights = torch.softmax(scaled_dot, dim=-1)

        if self.training:  # only add noise during training
            noise = torch.randn_like(attention_weights) * 0.01  # stddev can be tuned
            attention_weights = attention_weights + noise
            attention_weights = torch.clamp(attention_weights, min=0)  # prevent negatives
            attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)  # re-normalize

        return attention_weights

    @property
    def layers(self):
        return self.__layers


class SelfAttention:
    def __init__(self, input_shape, patch_size, patch_stride, transformer_d, top_k, direction):
        self.__input_height, self.__input_width, self.__input_channels = input_shape
        self.__patch_size, self.__patch_channels = patch_size, self.__input_channels
        self.__patch_stride = patch_stride
        self.__input_d = self.__get_input_dimension()

        self.__top_k = top_k
        self.__screen_dir = direction
        self.__num_patches = self.__get_number_of_patches()
        self.__patch_centers = self.__get_patch_centers()

        # Adjust input dimension to add 2 for positional encoding (y, x)
        self.__input_d_with_pos = self.__input_d + 2

        self.__attention = SelfAttentionModule(self.__input_d_with_pos, transformer_d)

        self.__transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.__input_height, self.__input_width)),
            transforms.ToTensor(),
        ])

    def __get_input_dimension(self):
        return self.__patch_channels * self.__patch_size ** 2

    def __get_number_of_patches(self):
        return int(((self.__input_height - self.__patch_size) / self.__patch_stride) + 1) * int(
            ((self.__input_width - self.__patch_size) / self.__patch_stride) + 1)

    def __get_patch_centers(self):
        offset = self.__patch_size // 2
        patch_centers = []
        n_row = int(((self.__input_height - self.__patch_size) / self.__patch_stride) + 1)
        n_column = int(((self.__input_width - self.__patch_size) / self.__patch_stride) + 1)
        for i in range(n_row):
            patch_center_row = offset + i * self.__patch_stride
            for j in range(n_column):
                patch_center_col = offset + j * self.__patch_stride
                patch_centers.append([patch_center_row, patch_center_col])
        return torch.tensor(patch_centers).float()

    def __get_flatten_patches(self, raw_data):
        data = self.__transform(raw_data).permute(1, 2, 0)

        patches = data.unfold(0, self.__patch_size, self.__patch_stride).permute(0, 3, 1, 2)
        patches = patches.unfold(2, self.__patch_size, self.__patch_stride).permute(0, 2, 1, 4, 3)
        patches = patches.reshape((-1, self.__patch_size, self.__patch_size, self.__patch_channels))
        flattened_patches = patches.reshape((1, -1, self.__input_d))

        # Normalize patch centers to [0, 1]
        normalized_centers = self.__patch_centers / torch.tensor([self.__input_height, self.__input_width]).float()

        # Add positional encoding: concatenate normalized (y, x) coords to each patch embedding
        # flattened_patches shape: (1, num_patches, input_d)
        # normalized_centers shape: (num_patches, 2)
        pos_encoding = normalized_centers.unsqueeze(0)  # (1, num_patches, 2)

        patches_with_pos = torch.cat([flattened_patches, pos_encoding], dim=-1)  # (1, num_patches, input_d+2)
        return patches_with_pos

    @staticmethod
    def __apply_softmax_on_columns(input_matrix):
        patch_importance_matrix = torch.softmax(input_matrix.squeeze(), dim=-1)
        return patch_importance_matrix

    @staticmethod
    def __get_importance_vector(input_matrix):
        importance_vector = input_matrix.sum(dim=0)
        return importance_vector

    def __get_top_k_patches(self, importance_vector):
        ix = torch.argsort(importance_vector, descending=True)
        top_k_ix = ix[:self.__top_k]
        centers = self.__patch_centers
        centers = centers[top_k_ix]
        return centers

    def get_output(self, raw_data):
        flatten_patches = self.__get_flatten_patches(raw_data=raw_data)
        attention_matrix = self.__attention(flatten_patches)
        attention_matrix = self.__apply_softmax_on_columns(input_matrix=attention_matrix)
        importance_vector = self.__get_importance_vector(input_matrix=attention_matrix)
        top_k_patches = self.__get_top_k_patches(importance_vector)
        return top_k_patches

    def normalize_patch_centers(self, ob):
        new_ob = ob / torch.tensor([self.__input_height, self.__input_width])
        centers = np.array(new_ob.flatten())
        return centers

    @property
    def layers(self):
        return self.__attention.layers
