<<<<<<< HEAD
# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
from typing import Tuple, Type

import torch
from torch import Tensor, nn

from ultralytics.nn.modules import MLPBlock


class TwoWayTransformer(nn.Module):
    """
    A Two-Way Transformer module that enables the simultaneous attention to both image and query points. This class
    serves as a specialized transformer decoder that attends to an input image using queries whose positional embedding
    is supplied. This is particularly useful for tasks like object detection, image segmentation, and point cloud
    processing.

    Attributes:
        depth (int): The number of layers in the transformer.
        embedding_dim (int): The channel dimension for the input embeddings.
        num_heads (int): The number of heads for multihead attention.
        mlp_dim (int): The internal channel dimension for the MLP block.
        layers (nn.ModuleList): The list of TwoWayAttentionBlock layers that make up the transformer.
        final_attn_token_to_image (Attention): The final attention layer applied from the queries to the image.
        norm_final_attn (nn.LayerNorm): The layer normalization applied to the final queries.
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                ))

        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must have same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          (torch.Tensor): the processed point_embedding
          (torch.Tensor): the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    """
    An attention block that performs both self-attention and cross-attention in two directions: queries to keys and
    keys to queries. This block consists of four main layers: (1) self-attention on sparse inputs, (2) cross-attention
    of sparse inputs to dense inputs, (3) an MLP block on sparse inputs, and (4) cross-attention of dense inputs to
    sparse inputs.

    Attributes:
        self_attn (Attention): The self-attention layer for the queries.
        norm1 (nn.LayerNorm): Layer normalization following the first attention block.
        cross_attn_token_to_image (Attention): Cross-attention layer from queries to keys.
        norm2 (nn.LayerNorm): Layer normalization following the second attention block.
        mlp (MLPBlock): MLP block that transforms the query embeddings.
        norm3 (nn.LayerNorm): Layer normalization following the MLP block.
        norm4 (nn.LayerNorm): Layer normalization following the third attention block.
        cross_attn_image_to_token (Attention): Cross-attention layer from keys to queries.
        skip_first_layer_pe (bool): Whether to skip the positional encoding in the first layer.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse inputs, (2) cross attention of sparse
        inputs to dense inputs, (3) mlp block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Args:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply self-attention and cross-attention to queries and keys and return the processed embeddings."""

        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """An attention layer that allows for downscaling the size of the embedding after projection to queries, keys, and
    values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        """
        Initializes the Attention model with the given dimensions and settings.

        Args:
            embedding_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
            downsample_rate (int, optional): The factor by which the internal dimensions are downsampled. Defaults to 1.

        Raises:
            AssertionError: If 'num_heads' does not evenly divide the internal dimension (embedding_dim / downsample_rate).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, 'num_heads must divide embedding_dim.'

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    @staticmethod
    def _separate_heads(x: Tensor, num_heads: int) -> Tensor:
        """Separate the input tensor into the specified number of attention heads."""
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    @staticmethod
    def _recombine_heads(x: Tensor) -> Tensor:
        """Recombine the separated attention heads into a single tensor."""
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Compute the attention output given the input query, key, and value tensors."""

        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        return self.out_proj(out)
=======
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
from typing import Tuple, Type

import torch
from torch import Tensor, nn

from ultralytics.nn.modules import MLPBlock


class TwoWayTransformer(nn.Module):
    """
    A Two-Way Transformer module for simultaneous attention to image and query points.

    This class implements a specialized transformer decoder that attends to an input image using queries with
    supplied positional embeddings. It's useful for tasks like object detection, image segmentation, and point
    cloud processing.

    Attributes:
        depth (int): Number of layers in the transformer.
        embedding_dim (int): Channel dimension for input embeddings.
        num_heads (int): Number of heads for multihead attention.
        mlp_dim (int): Internal channel dimension for the MLP block.
        layers (nn.ModuleList): List of TwoWayAttentionBlock layers composing the transformer.
        final_attn_token_to_image (Attention): Final attention layer from queries to image.
        norm_final_attn (nn.LayerNorm): Layer normalization applied to final queries.

    Methods:
        forward: Process image and point embeddings through the transformer.

    Examples:
        >>> transformer = TwoWayTransformer(depth=6, embedding_dim=256, num_heads=8, mlp_dim=2048)
        >>> image_embedding = torch.randn(1, 256, 32, 32)
        >>> image_pe = torch.randn(1, 256, 32, 32)
        >>> point_embedding = torch.randn(1, 100, 256)
        >>> output_queries, output_image = transformer(image_embedding, image_pe, point_embedding)
        >>> print(output_queries.shape, output_image.shape)
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        Initialize a Two-Way Transformer for simultaneous attention to image and query points.

        Args:
            depth (int): Number of layers in the transformer.
            embedding_dim (int): Channel dimension for input embeddings.
            num_heads (int): Number of heads for multihead attention. Must divide embedding_dim.
            mlp_dim (int): Internal channel dimension for the MLP block.
            activation (Type[nn.Module], optional): Activation function to use in the MLP block.
            attention_downsample_rate (int, optional): Downsampling rate for attention mechanism.
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        point_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process image and point embeddings through the Two-Way Transformer.

        Args:
            image_embedding (torch.Tensor): Image to attend to, with shape (B, embedding_dim, H, W).
            image_pe (torch.Tensor): Positional encoding to add to the image, with same shape as image_embedding.
            point_embedding (torch.Tensor): Embedding to add to query points, with shape (B, N_points, embedding_dim).

        Returns:
            queries (torch.Tensor): Processed point embeddings with shape (B, N_points, embedding_dim).
            keys (torch.Tensor): Processed image embeddings with shape (B, H*W, embedding_dim).
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    """
    A two-way attention block for simultaneous attention to image and query points.

    This class implements a specialized transformer block with four main layers: self-attention on sparse inputs,
    cross-attention of sparse inputs to dense inputs, MLP block on sparse inputs, and cross-attention of dense
    inputs to sparse inputs.

    Attributes:
        self_attn (Attention): Self-attention layer for queries.
        norm1 (nn.LayerNorm): Layer normalization after self-attention.
        cross_attn_token_to_image (Attention): Cross-attention layer from queries to keys.
        norm2 (nn.LayerNorm): Layer normalization after token-to-image attention.
        mlp (MLPBlock): MLP block for transforming query embeddings.
        norm3 (nn.LayerNorm): Layer normalization after MLP block.
        norm4 (nn.LayerNorm): Layer normalization after image-to-token attention.
        cross_attn_image_to_token (Attention): Cross-attention layer from keys to queries.
        skip_first_layer_pe (bool): Whether to skip positional encoding in the first layer.

    Methods:
        forward: Apply self-attention and cross-attention to queries and keys.

    Examples:
        >>> embedding_dim, num_heads = 256, 8
        >>> block = TwoWayAttentionBlock(embedding_dim, num_heads)
        >>> queries = torch.randn(1, 100, embedding_dim)
        >>> keys = torch.randn(1, 1000, embedding_dim)
        >>> query_pe = torch.randn(1, 100, embedding_dim)
        >>> key_pe = torch.randn(1, 1000, embedding_dim)
        >>> processed_queries, processed_keys = block(queries, keys, query_pe, key_pe)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        Initialize a TwoWayAttentionBlock for simultaneous attention to image and query points.

        This block implements a specialized transformer layer with four main components: self-attention on sparse
        inputs, cross-attention of sparse inputs to dense inputs, MLP block on sparse inputs, and cross-attention
        of dense inputs to sparse inputs.

        Args:
            embedding_dim (int): Channel dimension of the embeddings.
            num_heads (int): Number of attention heads in the attention layers.
            mlp_dim (int, optional): Hidden dimension of the MLP block.
            activation (Type[nn.Module], optional): Activation function for the MLP block.
            attention_downsample_rate (int, optional): Downsampling rate for the attention mechanism.
            skip_first_layer_pe (bool, optional): Whether to skip positional encoding in the first layer.
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, query_pe: torch.Tensor, key_pe: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply two-way attention to process query and key embeddings in a transformer block.

        Args:
            queries (torch.Tensor): Query embeddings with shape (B, N_queries, embedding_dim).
            keys (torch.Tensor): Key embeddings with shape (B, N_keys, embedding_dim).
            query_pe (torch.Tensor): Positional encodings for queries with same shape as queries.
            key_pe (torch.Tensor): Positional encodings for keys with same shape as keys.

        Returns:
            queries (torch.Tensor): Processed query embeddings with shape (B, N_queries, embedding_dim).
            keys (torch.Tensor): Processed key embeddings with shape (B, N_keys, embedding_dim).
        """
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer with downscaling capability for embedding size after projection.

    This class implements a multi-head attention mechanism with the option to downsample the internal
    dimension of queries, keys, and values.

    Attributes:
        embedding_dim (int): Dimensionality of input embeddings.
        kv_in_dim (int): Dimensionality of key and value inputs.
        internal_dim (int): Internal dimension after downsampling.
        num_heads (int): Number of attention heads.
        q_proj (nn.Linear): Linear projection for queries.
        k_proj (nn.Linear): Linear projection for keys.
        v_proj (nn.Linear): Linear projection for values.
        out_proj (nn.Linear): Linear projection for output.

    Methods:
        _separate_heads: Separate input tensor into attention heads.
        _recombine_heads: Recombine separated attention heads.
        forward: Compute attention output for given query, key, and value tensors.

    Examples:
        >>> attn = Attention(embedding_dim=256, num_heads=8, downsample_rate=2)
        >>> q = torch.randn(1, 100, 256)
        >>> k = v = torch.randn(1, 50, 256)
        >>> output = attn(q, k, v)
        >>> print(output.shape)
        torch.Size([1, 100, 256])
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        kv_in_dim: int = None,
    ) -> None:
        """
        Initialize the Attention module with specified dimensions and settings.

        Args:
            embedding_dim (int): Dimensionality of input embeddings.
            num_heads (int): Number of attention heads.
            downsample_rate (int, optional): Factor by which internal dimensions are downsampled.
            kv_in_dim (int | None, optional): Dimensionality of key and value inputs. If None, uses embedding_dim.

        Raises:
            AssertionError: If num_heads does not evenly divide the internal dim (embedding_dim / downsample_rate).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    @staticmethod
    def _separate_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Separate the input tensor into the specified number of attention heads."""
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    @staticmethod
    def _recombine_heads(x: Tensor) -> Tensor:
        """Recombine separated attention heads into a single tensor."""
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention to query, key, and value tensors with optional downsampling.

        Args:
            q (torch.Tensor): Query tensor with shape (B, N_q, embedding_dim).
            k (torch.Tensor): Key tensor with shape (B, N_k, embedding_dim).
            v (torch.Tensor): Value tensor with shape (B, N_k, embedding_dim).

        Returns:
            (torch.Tensor): Output tensor after attention with shape (B, N_q, embedding_dim).
        """
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        return self.out_proj(out)
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
