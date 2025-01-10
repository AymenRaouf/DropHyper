import dhg
import torch
import torch.nn as nn
from dhg.structure.graphs import Graph
from dhg.nn import HGNNPConv, HGNNConv, HyperGCNConv, UniGCNConv, UniGATConv, UniSAGEConv, UniGINConv, MultiHeadWrapper, HNHNConv



class deepHGNN(nn.Module):
    r"""Deep version of the HGNN model proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``depth`` (``int``): The Number of layers of the model that conduct message passing (final prediction layer not included).
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        depth: int = 1,
        use_bn: bool = False,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                HGNNConv(in_channels, in_channels, use_bn=use_bn, drop_rate=drop_rate)
            )
        self.layers.append(
            HGNNConv(in_channels, num_classes, use_bn=use_bn, is_last=True)
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return torch.nn.Softmax(X)




class deepHGNNP(nn.Module):
    r"""Deep version of the HGNN :sup:`+` model proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``depth`` (``int``): The Number of layers of the model that conduct message passing (final prediction layer not included).
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.0``.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        depth: int = 1,
        use_bn: bool = False,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                HGNNPConv(in_channels, in_channels, use_bn=use_bn, drop_rate=drop_rate)
            )
        self.layers.append(
            HGNNPConv(in_channels, num_classes, use_bn=use_bn, is_last=True)
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X
    


class deepHyperGCN(nn.Module):
    r"""Deep version of the HyperGCN model proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://papers.nips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf>`_ paper (NeurIPS 2019).
    
    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``depth`` (``int``): The Number of layers of the model that conduct message passing (final prediction layer not included).
        ``use_mediator`` (``str``): Whether to use mediator to transform the hyperedges to edges in the graph. Defaults to ``False``.
        ``fast`` (``bool``): If set to ``True``, the transformed graph structure will be computed once from the input hypergraph and vertex features, and cached for future use. Defaults to ``True``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        depth: int = 1,
        use_mediator: bool = False,
        use_bn: bool = False,
        fast: bool = True,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.fast = fast
        self.cached_g = None
        self.with_mediator = use_mediator
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                HyperGCNConv(
                    in_channels, in_channels, use_mediator, use_bn=use_bn, drop_rate=drop_rate,
                )
            )
        self.layers.append(
            HyperGCNConv(
                in_channels, num_classes, use_mediator, use_bn=use_bn, is_last=True
            )
        )


    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        if self.fast:
            if self.cached_g is None:
                self.cached_g = Graph.from_hypergraph_hypergcn(
                    hg, X, self.with_mediator
                )
            for layer in self.layers:
                X = layer(X, hg, self.cached_g)
        else:
            for layer in self.layers:
                X = layer(X, hg)
        return X

    

class deepUniGCN(nn.Module):
    r"""Deep version of the UniGCN model proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``depth`` (``int``): The Number of layers of the model that conduct message passing (final prediction layer not included).
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.0``.
    """

    def __init__(
        self, 
        in_channels: int, 
        num_classes: int, 
        depth: int = 1, 
        use_bn: bool = False, 
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                UniGCNConv(
                    in_channels, in_channels, use_bn=use_bn, drop_rate=drop_rate
                )
            )
        self.layers.append(UniGCNConv(in_channels, num_classes, use_bn=use_bn, is_last=True))

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X



class deepUniGAT(nn.Module):
    r"""Deep version of the UniGAT model proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``num_heads`` (``int``): The Number of attention head in each layer.
        ``depth`` (``int``): The Number of layers of the model that conduct message passing (final prediction layer not included).
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): The dropout probability. Defaults to ``0.0``.
        ``atten_neg_slope`` (``float``): Hyper-parameter of the ``LeakyReLU`` activation of edge attention. Defaults to 0.2.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_heads: int,
        depth: int = 1, 
        use_bn: bool = False,
        drop_rate: float = 0.0,
        atten_neg_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.drop_layer = nn.Dropout(drop_rate)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                MultiHeadWrapper(
                    num_heads,
                    "mean",
                    UniGATConv,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    use_bn=use_bn,
                    drop_rate=drop_rate,
                    atten_neg_slope=atten_neg_slope,
                )
            )
            # The original implementation has applied activation layer after the final layer.
            # Thus, we donot set ``is_last`` to ``True``.
        self.out_layer = UniGATConv(
            in_channels,
            num_classes,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
            is_last=False,
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = self.drop_layer(X)
            X = layer(X=X, hg=hg)
            X = self.drop_layer(X)
        X = self.out_layer(X=X, hg=hg)
        return X



class deepUniSAGE(nn.Module):
    r"""Deep version of the UniSAGE model proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``depth`` (``int``): The Number of layers of the model that conduct message passing (final prediction layer not included).
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.0``.
    """

    def __init__(
        self, 
        in_channels: int, 
        num_classes: int,
        depth: int = 1,
        use_bn: bool = False, 
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                UniSAGEConv(
                    in_channels, in_channels, use_bn=use_bn, drop_rate=drop_rate
                )
            )
                
        self.layers.append(UniSAGEConv(in_channels, num_classes, use_bn=use_bn, is_last=True))

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return torch.sigmoid(X)



class deepUniGIN(nn.Module):
    r"""Deep version of the UniGIN model proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``depth`` (``int``): The Number of layers of the model that conduct message passing (final prediction layer not included).
        ``eps`` (``float``): The epsilon value. Defaults to ``0.0``.
        ``train_eps`` (``bool``): If set to ``True``, the epsilon value will be trainable. Defaults to ``False``.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.0``.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        depth: int = 1,
        eps: float = 0.0,
        train_eps: bool = False,
        use_bn: bool = False,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                UniGINConv(
                    in_channels, in_channels, eps=eps, train_eps=train_eps, use_bn=use_bn, drop_rate=drop_rate
                )
            )
        self.layers.append(
            UniGINConv(in_channels, num_classes, eps=eps, train_eps=train_eps, use_bn=use_bn, is_last=True)
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X


class deepHNHN(nn.Module):
    r"""Deep version of the HNHN model proposed in `HNHN: Hypergraph Networks with Hyperedge Neurons <https://arxiv.org/pdf/2006.12278.pdf>`_ paper (ICML 2020).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``depth`` (``int``): The Number of layers of the model that conduct message passing (final prediction layer not included).
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.0``.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        depth: int = 1,
        use_bn: bool = False,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                HNHNConv(
                    in_channels, in_channels, use_bn=use_bn, drop_rate=drop_rate
                )
            )
        self.layers.append(
            HNHNConv(in_channels, num_classes, use_bn=use_bn, is_last=True)
        )


    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X


def load_model(name, X, data, depth):
    match name:
        case 'HGNN':
            return deepHGNN(X.shape[1], data["num_classes"], depth)
        case 'HGNNP':
            return deepHGNNP(X.shape[1], data["num_classes"], depth)
        case 'HyperGCN':
            return deepHyperGCN(X.shape[1], data["num_classes"], depth)
        case 'UniSAGE':
            return deepUniSAGE(X.shape[1], data["num_classes"], depth)
        case 'UniGCN':
            return deepUniGCN(X.shape[1], data["num_classes"], depth)
        case 'UniGAT':
            return deepUniGAT(X.shape[1], data["num_classes"], 2, depth)
        case 'UniGIN':
            return deepUniGIN(X.shape[1], data["num_classes"], depth)
        case 'HNHN':
            return deepHNHN(X.shape[1], data["num_classes"], depth)
