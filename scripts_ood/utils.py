import os
import torch
import numpy as np
from tqdm import tqdm
import torchvision.models as models
import torchvision
import shutil
from PIL import Image


def save_with_check(save_path, array, verbose):
    if os.path.isfile(save_path):
        if verbose:
            print(f"{save_path} already exists")
    else:
        np.save(save_path, array)

def maybe_print(s: str, flag: bool):
    if flag:
        print(s)

def load_pretrained_weights_vit(
    model, 
    model_name=None, 
    weights_path=None, 
    load_first_conv=True, 
    load_fc=True, 
    load_repr_layer=False,
    resize_positional_embedding=False,
    verbose=True,
    strict=True,
):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): Full model (a nn.Module)
        model_name (str): Model name (e.g. B_16)
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_first_conv (bool): Whether to load patch embedding.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        resize_positional_embedding=False,
        verbose (bool): Whether to print on completion
    """
    assert bool(model_name) ^ bool(weights_path), 'Expected exactly one of model_name or weights_path'
    
    # Load or download weights
    if weights_path is None:
        url = PRETRAINED_MODELS[model_name]['url']
        if url:
            state_dict = model_zoo.load_url(url)
        else:
            raise ValueError(f'Pretrained model for {model_name} has not yet been released')
    else:
        state_dict = torch.load(weights_path)

    # Modifications to load partial state dict
    expected_missing_keys = []
    if not load_first_conv and 'patch_embedding.weight' in state_dict:
        expected_missing_keys += ['patch_embedding.weight', 'patch_embedding.bias']
    if not load_fc and 'fc.weight' in state_dict:
        expected_missing_keys += ['fc.weight', 'fc.bias']
    if not load_repr_layer and 'pre_logits.weight' in state_dict:
        expected_missing_keys += ['pre_logits.weight', 'pre_logits.bias']
    for key in expected_missing_keys:
        state_dict.pop(key)

    # Change size of positional embeddings
    if resize_positional_embedding: 
        posemb = state_dict['positional_embedding.pos_embedding']
        posemb_new = model.state_dict()['positional_embedding.pos_embedding']
        state_dict['positional_embedding.pos_embedding'] = \
            resize_positional_embedding_(posemb=posemb, posemb_new=posemb_new, 
                has_class_token=hasattr(model, 'class_token'))
        maybe_print('Resized positional embeddings from {} to {}'.format(
                    posemb.shape, posemb_new.shape), verbose)

    # Load state dict
    ret = model.load_state_dict(state_dict, strict=False)
    if strict:
        assert set(ret.missing_keys) == set(expected_missing_keys), \
            'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
        assert not ret.unexpected_keys, \
            'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)
        maybe_print('Loaded pretrained weights.', verbose)
    else:
        maybe_print('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys), verbose)
        maybe_print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys), verbose)
        return ret

def create_9_vs_1_cifar_emb(model_name, ood_label, verbose=False):
    train_path = f"../data/predictions/{model_name}_cifar_train.npy"
    test_path = f"../data/predictions/{model_name}_cifar_test.npy"
    train = np.load(train_path)
    test = np.load(test_path)

    train_in = train[train[:, -1] != ood_label]
    test_in = test[test[:, -1] != ood_label]
    test_out = test[test[:, -1] == ood_label]

    train_in_path = f"../data/predictions/ood_vs_class/{model_name}_cifar_train_without_{ood_label}_class.npy"
    test_in_path = f"../data/predictions/ood_vs_class/{model_name}_cifar_test_without_{ood_label}_class.npy"
    test_out_path = f"../data/predictions/ood_vs_class/{model_name}_cifar_test_only_{ood_label}_class.npy"

    save_with_check(train_in_path, train_in, verbose)
    save_with_check(test_in_path, test_in, verbose)
    save_with_check(test_out_path, test_out, verbose)

    return train_in_path, test_in_path, test_out_path


def imagenet_sanity_check(model, transform, device):
    names = []
    with open("/workspaces/ood/data/imagenet1000_clsidx_to_labels.txt") as fd:
        for line in fd:
            name = line[line.find(" ") :][1:-2]
            names.append(name)

    cat = Image.open("/workspaces/ood/data/test_samples/cat.jpg")
    cat = cat.convert("RGB")
    pred = (
        model(torch.unsqueeze(transform(cat).to(device), dim=0)).cpu().detach().numpy()
    )
    a = np.argsort(pred[0])[::-1]
    print([names[x] for x in a[:10]])


def create_mvtec_dataset():
    download = False
    cifar_data_train = torchvision.datasets.CIFAR10(
        "../data/cifar10", download=download
    )
    cifar_data_test = torchvision.datasets.CIFAR10(
        "../data/cifar10", download=download, train=False
    )

    svhn_data_train = torchvision.datasets.SVHN("../data/svhn", download=download)
    svhn_data_test = torchvision.datasets.SVHN(
        "../data/svhn", download=download, split="test"
    )
    ds_path = "/workspaces/ood/data/cifar10_full_size/ood"
    shutil.rmtree(ds_path, ignore_errors=True)

    os.makedirs(os.path.join(ds_path, "train/good/"))
    os.makedirs(os.path.join(ds_path, "test/good/"))
    os.makedirs(os.path.join(ds_path, "test/anomaly/"))

    max_count = 5000

    label_counts = {i: 0 for i in range(10)}
    for i, (image, label) in tqdm(enumerate(cifar_data_train)):
        label_counts[label] += 1
        if label_counts[label] > max_count:
            continue
        image.save(
            f"{ds_path}/train/good/{i}_{label}.png",
            "PNG",
        )

    max_count = 100000
    label_counts = {i: 0 for i in range(10)}
    for i, (image, label) in tqdm(enumerate(cifar_data_test)):
        label_counts[label] += 1
        if label_counts[label] > max_count:
            continue
        image.save(
            f"{ds_path}/test/good/{i}_{label}.png",
            "PNG",
        )

    max_count = 100000
    label_counts = {i: 0 for i in range(10)}
    for i, (image, label) in tqdm(enumerate(svhn_data_test)):
        label_counts[label] += 1
        if label_counts[label] > max_count:
            continue
        image.save(
            f"{ds_path}/test/anomaly/{i}_{label}.png",
            "PNG",
        )


def load_byol(check_point_path):
    model = models.__dict__["resnet50"]()
    for name, param in model.named_parameters():
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False

    state_dict = torch.load(check_point_path)
    model.load_state_dict(state_dict, strict=False)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model


def load_moco(check_point_path, device):
    model = models.__dict__["resnet50"]()

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    checkpoint = torch.load(check_point_path, map_location="cuda")

    # rename moco pre-trained keys
    state_dict = checkpoint["state_dict"]
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
            # remove prefix
            state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.to(device)
    return model


def prettify_value(numbers, num_round, ljust_num):

    return (
        str(np.round(number, num_round)).ljust(ljust_num, "0") for number in numbers
    )


def predict_on_whole_dataset(model, dataset, out_name, device):
    out_dir = f"/workspaces/ood/data/predictions/{out_name}.npy"
    if os.path.isfile(out_dir):
        print(f"Predictions are already present in {out_dir}")
        return
    model.eval()
    preds = []
    image_clses = []

    i = 0
    with torch.no_grad():
        for image, image_cls in tqdm(dataset):
            # if i == 100:
            #     break
            # i+=1
            pred = model(torch.unsqueeze(image.to(device), dim=0))
            preds.append(pred.cpu().detach())
            image_clses.append(image_cls)

    image_clses = torch.unsqueeze(torch.tensor(np.array(image_clses)), -1)
    model_pred = torch.cat([torch.cat(preds), image_clses], -1).cpu().detach().numpy()
    np.save(out_dir, model_pred)


def add_labels(dataset, out_name):
    out_dir = f"/workspaces/ood/data/predictions/{out_name}.npy"

    data = np.load(out_dir)
    if data.shape[1] > 1000:
        print("Has labels!")
        return
    labels = []
    for _, image_cls in tqdm(dataset):
        labels.append(image_cls)
    data_with_labels = np.concatenate(
        [data, np.expand_dims(np.array(labels), -1)], axis=1
    )
    np.save(out_dir, data_with_labels)
