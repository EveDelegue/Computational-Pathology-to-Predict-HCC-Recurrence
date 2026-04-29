import joblib
import numpy as np
from scipy.spatial import distance
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
import matplotlib.pyplot as plt

from scipy.ndimage import label
from scipy import ndimage
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import watershed

import cv2
from termcolor import colored
import os
from tqdm import tqdm

# defining the coloring dictionary: a dictionary that specifies a color to each class {type_id : (type_name, colour)}
color_dict_pannuke = {
    0: ("background", (255, 255, 0)),  # YELLOW
    1: ("neoplastic epithelial", (255, 255, 0)),  # YELLOW
    2: ("Inflammatory", (0, 191, 255)),  # RED
    3: ("Connective", (255, 255, 0)),  # YELLOW
    4: ("Dead", (255, 255, 0)),  # YELLOW
    5: ("non-neoplastic epithelial", (255, 255, 0)),  # YELLOW
}

color_dict_monusac = {
    0: ("Epithelial", (255, 255, 0)),  # YELLOW
    1: ("Lymphocyte", (0, 191, 255)),  # green
    2: ("Macrophage", (0, 191, 255)),  # green
    3: ("Neutrophil", (255, 255, 0)),  # YELLOW
    4: ("Inflammatory", (0, 191, 255)),  # RED
}

color_dict_consep = {
    0: ("background", (255, 255, 0)),  # YELLOW
    1: ("Epithelial", (255, 255, 0)),  # YELLOW
    2: ("Inflammatory", (0, 191, 255)),  # RED
    3: ("Spindle-Shaped", (255, 255, 0)),  # YELLOW
    4: ("Miscellaneous", (255, 255, 0)),  # YELLOW
}

color_dict_AllInflams = {
    1: ("Lymphocyte", (0, 191, 255)),  # RED
    4: ("Inflammatory", (0, 191, 255)),
    2: ("Macrophage", (0, 191, 255)),
}


def get_Inflammatory(tile_preds, color_dict):
    """
    This function filters out nuclei that are classified as either 'Inflammatory' or 'Lymphocyte' based on the given
    tile predictions and color dictionary. It also counts the number of such nuclei.

    Parameters:
    tile_preds (dict): A dictionary containing nucleus predictions. Each key is a nucleus ID, and the corresponding value is
                        another dictionary containing the nucleus's properties, including its type.
    color_dict (dict): A dictionary mapping class labels to their corresponding descriptions and colors.

    Returns:
    tuple: A tuple containing two elements:
        - Inflammatory_nucleus (dict): A dictionary containing only the nuclei that are classified as 'Inflammatory' or
                                    'Lymphocyte'. Each key is a nucleus ID, and the corresponding value is the nucleus's
                                    properties.
        - number (int): The total number of 'Inflammatory' or 'Lymphocyte' nuclei found.
    """
    number = 0
    Inflammatory_nucleus = {}
    for nuc_id in tile_preds:
        class_label = tile_preds[nuc_id]["type"]
        classe = color_dict[class_label][0]
        if classe in ["Inflammatory", "Lymphocyte", "Macrophage"]:
            number += 1
            Inflammatory_nucleus[nuc_id] = tile_preds[nuc_id]
    return Inflammatory_nucleus, number


def filter_centroids(centroids, d_threshold=1.5):
    # Define distance threshold
    d_threshold = 1.5  # Adjust this value based on your data
    # List to store the filtered centroids
    filtered_centroids = []
    # Flag array to mark centroids that have already been clustered
    visited = np.zeros(len(centroids), dtype=bool)
    for i, centroid in enumerate(tqdm(centroids)):
        if visited[i]:
            continue
        # Group of centroids that are within d_threshold of the current centroid
        cluster = [centroid]
        # Mark the current centroid as visited
        visited[i] = True
        for j, other_centroid in enumerate(centroids[i + 1 :], start=i + 1):
            if (
                not visited[j]
                and distance.euclidean(centroid, other_centroid) <= d_threshold
            ):
                cluster.append(other_centroid)
                visited[j] = True
        # Choose the "big" centroid to keep; here we take the one with the largest sum of coordinates
        # You can adjust this to your criteria, like averaging the coordinates instead
        big_centroid = max(cluster, key=lambda x: sum(x))
        filtered_centroids.append(big_centroid)
    return filtered_centroids


def detect_Inflammatory_cells(img_path, models, colors):
    dict4 = {}
    for model, color in zip(models, colors):
        tile_output = model.predict([img_path], mode="tile")
        tile_preds = joblib.load(f"{tile_output[0][1]}.dat")
        Inflammatory_nucleus1, _ = get_Inflammatory(tile_preds, color)
        dict4.update(Inflammatory_nucleus1)
    nuc_centroids = {
        tuple(np.uint8(info["centroid"])): nuc_id for nuc_id, info in dict4.items()
    }

    filtered_centroids = np.array(
        filter_centroids(np.array(list(nuc_centroids.keys())), d_threshold=1.5)
    )
    filtred_dict = {}
    for centroid in filtered_centroids:
        nuc_id = nuc_centroids[tuple(centroid)]
        filtred_dict[nuc_id] = dict4[nuc_id]
    return filtred_dict


################### functions from https://github.com/vqdang/hover_net by 
# @article{graham2019hover,
#  title={Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images},
#  author={Graham, Simon and Vu, Quoc Dang and Raza, Shan E Ahmed and Azam, Ayesha and Tsang, Yee Wah and Kwak, Jin Tae and Rajpoot, Nasir},
#  journal={Medical Image Analysis},
#  pages={101563},
#  year={2019},
#  publisher={Elsevier}
#}



def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                "%s: Detect checkpoint saved in data-parallel mode."
                " Converting saved model to single GPU mode." % colored_word
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict

def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image.

    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`
        
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x

def crop_to_shape(x, y, data_format="NCHW"):
    """Centre crop x so that x has shape of y. y dims must be smaller than x dims.

    Args:
        x: input array
        y: array with desired shape.

    """
    assert (
        y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1]
    ), "Ensure that y dimensions are smaller than x dimensions!"

    x_shape = x.size()
    y_shape = y.size()
    if data_format == "NCHW":
        crop_shape = (x_shape[2] - y_shape[2], x_shape[3] - y_shape[3])
    else:
        crop_shape = (x_shape[1] - y_shape[1], x_shape[2] - y_shape[2])
    return crop_op(x, crop_shape, data_format)

####
class Net(nn.Module):
    """ A base class provides a common weight initialisation scheme."""

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            # ! Fixed the type checking
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            if "norm" in classname.lower():
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if "linear" in classname.lower():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return x

####
class TFSamepaddingLayer(nn.Module):
    """To align with tf `same` padding. 
    
    Putting this before any conv layer that need padding
    Assuming kernel has Height == Width for simplicity
    """

    def __init__(self, ksize, stride):
        super(TFSamepaddingLayer, self).__init__()
        self.ksize = ksize
        self.stride = stride

    def forward(self, x):
        if x.shape[2] % self.stride == 0:
            pad = max(self.ksize - self.stride, 0)
        else:
            pad = max(self.ksize - (x.shape[2] % self.stride), 0)

        if pad % 2 == 0:
            pad_val = pad // 2
            padding = (pad_val, pad_val, pad_val, pad_val)
        else:
            pad_val_start = pad // 2
            pad_val_end = pad - pad_val_start
            padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end)
        # print(x.shape, padding)
        x = F.pad(x, padding, "constant", 0)
        # print(x.shape)
        return x

####
class DenseBlock(Net):
    """Dense Block as defined in:

    Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. 
    "Densely connected convolutional networks." In Proceedings of the IEEE conference 
    on computer vision and pattern recognition, pp. 4700-4708. 2017.

    Only performs `valid` convolution.

    """

    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, split=1):
        super(DenseBlock, self).__init__()
        assert len(unit_ksize) == len(unit_ch), "Unbalance Unit Info"

        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch

        # ! For inference only so init values for batchnorm may not match tensorflow
        unit_in_ch = in_ch
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            self.units.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("preact_bna/bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                            ("preact_bna/relu", nn.ReLU(inplace=True)),
                            (
                                "conv1",
                                nn.Conv2d(
                                    unit_in_ch,
                                    unit_ch[0],
                                    unit_ksize[0],
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                ),
                            ),
                            ("conv1/bn", nn.BatchNorm2d(unit_ch[0], eps=1e-5)),
                            ("conv1/relu", nn.ReLU(inplace=True)),
                            # ('conv2/pool', TFSamepaddingLayer(ksize=unit_ksize[1], stride=1)),
                            (
                                "conv2",
                                nn.Conv2d(
                                    unit_ch[0],
                                    unit_ch[1],
                                    unit_ksize[1],
                                    groups=split,
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                ),
                            ),
                        ]
                    )
                )
            )
            unit_in_ch += unit_ch[1]

        self.blk_bna = nn.Sequential(
            OrderedDict(
                [
                    ("bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

    def out_ch(self):
        return self.in_ch + self.nr_unit * self.unit_ch[-1]

    def forward(self, prev_feat):
        for idx in range(self.nr_unit):
            new_feat = self.units[idx](prev_feat)
            prev_feat = crop_to_shape(prev_feat, new_feat)
            prev_feat = torch.cat([prev_feat, new_feat], dim=1)
        prev_feat = self.blk_bna(prev_feat)

        return prev_feat

####
class ResidualBlock(Net):
    """Residual block as defined in:

    He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep residual learning 
    for image recognition." In Proceedings of the IEEE conference on computer vision 
    and pattern recognition, pp. 770-778. 2016.

    """

    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, stride=1):
        super(ResidualBlock, self).__init__()
        assert len(unit_ksize) == len(unit_ch), "Unbalance Unit Info"

        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch

        # ! For inference only so init values for batchnorm may not match tensorflow
        unit_in_ch = in_ch
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            unit_layer = [
                ("preact/bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                ("preact/relu", nn.ReLU(inplace=True)),
                (
                    "conv1",
                    nn.Conv2d(
                        unit_in_ch,
                        unit_ch[0],
                        unit_ksize[0],
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                ),
                ("conv1/bn", nn.BatchNorm2d(unit_ch[0], eps=1e-5)),
                ("conv1/relu", nn.ReLU(inplace=True)),
                (
                    "conv2/pad",
                    TFSamepaddingLayer(
                        ksize=unit_ksize[1], stride=stride if idx == 0 else 1
                    ),
                ),
                (
                    "conv2",
                    nn.Conv2d(
                        unit_ch[0],
                        unit_ch[1],
                        unit_ksize[1],
                        stride=stride if idx == 0 else 1,
                        padding=0,
                        bias=False,
                    ),
                ),
                ("conv2/bn", nn.BatchNorm2d(unit_ch[1], eps=1e-5)),
                ("conv2/relu", nn.ReLU(inplace=True)),
                (
                    "conv3",
                    nn.Conv2d(
                        unit_ch[1],
                        unit_ch[2],
                        unit_ksize[2],
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                ),
            ]
            # * has bna to conclude each previous block so
            # * must not put preact for the first unit of this block
            unit_layer = unit_layer if idx != 0 else unit_layer[2:]
            self.units.append(nn.Sequential(OrderedDict(unit_layer)))
            unit_in_ch = unit_ch[-1]

        if in_ch != unit_ch[-1] or stride != 1:
            self.shortcut = nn.Conv2d(in_ch, unit_ch[-1], 1, stride=stride, bias=False)
        else:
            self.shortcut = None

        self.blk_bna = nn.Sequential(
            OrderedDict(
                [
                    ("bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

        # print(self.units[0])
        # print(self.units[1])
        # exit()

    def out_ch(self):
        return self.unit_ch[-1]

    def forward(self, prev_feat, freeze=False):
        if self.shortcut is None:
            shortcut = prev_feat
        else:
            shortcut = self.shortcut(prev_feat)

        for idx in range(0, len(self.units)):
            new_feat = prev_feat
            if self.training:
                with torch.set_grad_enabled(not freeze):
                    new_feat = self.units[idx](new_feat)
            else:
                new_feat = self.units[idx](new_feat)
            prev_feat = new_feat + shortcut
            shortcut = prev_feat
        feat = self.blk_bna(prev_feat)
        return feat

####
class UpSample2x(nn.Module):
    """Upsample input by a factor of 2.
    
    Assume input is of NCHW, port FixedUnpooling from TensorPack.
    """

    def __init__(self):
        super(UpSample2x, self).__init__()
        # correct way to create constant within module
        self.register_buffer(
            "unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32"))
        )
        self.unpool_mat.unsqueeze(0)

    def forward(self, x):
        input_shape = list(x.shape)
        # unsqueeze is expand_dims equivalent
        # permute is transpose equivalent
        # view is reshape equivalent
        x = x.unsqueeze(-1)  # bchwx1
        mat = self.unpool_mat.unsqueeze(0)  # 1xshxsw
        ret = torch.tensordot(x, mat, dims=1)  # bxcxhxwxshxsw
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        ret = ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))
        return ret

####
class HoVerNet(Net):
    """Initialise HoVer-Net."""

    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        assert mode == 'original' or mode == 'fast', \
                'Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`.' % mode

        module_list = [
            ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        if mode == 'fast': # prepend the padding for `fast` mode
            module_list = [("pad", TFSamepaddingLayer(ksize=7, stride=1))] + module_list

        self.conv0 = nn.Sequential(OrderedDict(module_list))
        self.d0 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 3, stride=1)
        self.d1 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 4, stride=2)
        self.d2 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 6, stride=2)
        self.d3 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 3, stride=2)

        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            module_list = [ 
                ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False),),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0),])
            )
            return decoder

        ksize = 5 if mode == 'original' else 3
        if nr_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize,out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize,out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=nr_types)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )

        self.upsample2x = UpSample2x()
        # TODO: pytorch still require the channel eventhough its ignored
        self.weights_init()

    def forward(self, imgs):

        imgs = imgs / 255.0  # to 0-1 range to match XY

        if self.training:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0, self.freeze)
            with torch.set_grad_enabled(not self.freeze):
                d1 = self.d1(d0)
                d2 = self.d2(d1)
                d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]
        else:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0)
            d1 = self.d1(d0)
            d2 = self.d2(d1)
            d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]

        # TODO: switch to `crop_to_shape` ?
        if self.mode == 'original':
            d[0] = crop_op(d[0], [184, 184])
            d[1] = crop_op(d[1], [72, 72])
        else:
            d[0] = crop_op(d[0], [92, 92])
            d[1] = crop_op(d[1], [36, 36])

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict

class PatchDataset(Dataset):
    def __init__(self,dir):
        super().__init__()
        self.dir = dir
        self.list_dir = os.listdir(dir)
        self.transform = torchvision.transforms.PILToTensor()
    def __len__(self):
        return len(self.list_dir)
    
    def __getitem__(self, index):
        # read the image
        img_path = self.list_dir[index]
        image = Image.open(os.path.join(self.dir,img_path))
        _, _, x, _, y = img_path[:-4].split("_")
        return self.transform(image),int(x),int(y)

def load_net(device):
    net = HoVerNet(nr_types=6,mode='fast')
    if not os.path.exists(os.path.join(os.curdir,'models','hovernet_fast_pannuke_type_tf2pytorch.tar')):
        print('No pretrained NN. Download it in from https://drive.google.com/file/d/1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR/view \n and copy it to ',os.path.join(os.curdir,'models','hovernet_fast_pannuke_type_tf2pytorch.tar'))
    saved_state_dict = torch.load(os.path.join(os.curdir,'models','hovernet_fast_pannuke_type_tf2pytorch.tar'),)["desc"]
    saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)


    net.load_state_dict(saved_state_dict, strict=True)
    net = torch.nn.DataParallel(net)
    return net.to(device)

def inference(dataloader,net):
        raw_dict = {}
        for i,(imgs,xs,ys) in tqdm(enumerate(dataloader),total=len(dataloader)):
            pred_dict = net(imgs)
            pred_dict = OrderedDict(
                [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
            )
            pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1]
            if net.module.nr_types is not None:
                type_map = F.softmax(pred_dict["tp"], dim=-1)
                type_map = torch.argmax(type_map, dim=-1, keepdim=False)
                type_map = type_map.type(torch.float32)
                pred_dict["tp"] = type_map

            result_dict = { 
            "raw": {"xs":xs, "ys":ys,
            "prob_np": pred_dict["np"].cpu().numpy(),
            "pred_hv": pred_dict["hv"].cpu().numpy(),
            }
            }
            if net.module.nr_types is not None:
                result_dict["raw"]["pred_tp"] = pred_dict["tp"].cpu().numpy()
            raw_dict[i] = result_dict
        return raw_dict

# post processing
def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided. 

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel. 
    
    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def __proc_np_hv(blb_raw,h_dir_raw,v_dir_raw):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming 
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map

    """

    blb_raw = np.array(blb_raw, dtype=np.float32)
    h_dir_raw = np.array(h_dir_raw, dtype=np.float32)
    v_dir_raw = np.array(v_dir_raw, dtype=np.float32)

    ## processing
    # threshold
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)

    # get a label for each independant component
    blb = label(blb)[0]
    # remove small components
    blb = remove_small_objects(blb, min_size=10)
    # get a binary image back
    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    # get a map
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, dx=1, dy=0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, dx=0, dy=1, ksize=21)

    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    ) if sobelh.max()!= sobelh.min() else 0*sobelh

    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    ) if sobelv.max()!= sobelv.min() else 0*sobelv

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)
    # threshold
    overall = np.array(overall >= 0.4, dtype=np.int32)

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    # get ids
    marker = label(marker)[0]
    # remove small objects
    marker = remove_small_objects(marker, min_size=10)
    # watershed
    proced_pred = watershed(dist, markers=marker, mask=blb)

    return proced_pred


def get_n_inflam(post_processed, pred_tp):
    post_processed = post_processed.squeeze()
    instances = np.unique(post_processed)
    n_inflam = 0
    for inst in instances[1:]:
        inst_img = np.zeros_like(post_processed)
        inst_img[post_processed==inst] = 1
        rmin, rmax, cmin, cmax = get_bounding_box(inst_img)
        inst_types, nb_types = np.unique(pred_tp[..., rmin:rmax, cmin:cmax],return_counts=True)
        # get best type
        idx = np.argmax(nb_types)
        inst_type = inst_types[idx]
        if inst_type==0: # if backgroung
            # get second best
            nb_types[idx] = 0
            idx = np.argmax(nb_types)
            inst_type = inst_types[idx]
        
        if inst_type == 2:
            n_inflam+=1
    return n_inflam


def post_process(raw): 
    num_nucleus, coords_x, coords_y = [],[],[]
    # for each batch
    for _,batch_result in tqdm(raw.items()):
        prob_np,pred_hv,pred_tp,xs,ys = batch_result["raw"]['prob_np'], batch_result["raw"]['pred_hv'], batch_result["raw"]['pred_tp'], batch_result["raw"]['xs'], batch_result["raw"]['ys'] 
        # for each sample from a batch
        for i in range(prob_np.shape[0]):
            blb_raw = prob_np[i]
            h_dir_raw = pred_hv[i,...,0]
            v_dir_raw = pred_hv[i,...,1]
            x,y = int(xs[i]), int(ys[i])
            # post process
            post_processed = __proc_np_hv(blb_raw,h_dir_raw,v_dir_raw)
            n_inflam = get_n_inflam(post_processed,pred_tp)
            coords_x.append(x)
            coords_y.append(y)
            num_nucleus.append(n_inflam)
    return num_nucleus,coords_x,coords_y
            
