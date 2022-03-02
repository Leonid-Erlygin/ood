import os
import torch
import numpy as np
from tqdm import tqdm
import torchvision.models as models


def load_byol(check_point_path, device):
    model = models.__dict__["resnet50"]()
    for name, param in model.named_parameters():
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False

    state_dict = torch.load(check_point_path)
    model.load_state_dict(state_dict, strict=False)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.to(device)
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
    for image, image_cls in tqdm(dataset):
        pred = model(torch.unsqueeze(image.to(device), dim=0))
        preds.append(pred[:, :, 0, 0])
        image_clses.append(image_cls)

    image_clses = torch.unsqueeze(torch.tensor(np.array(image_clses)), -1).to(device)
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
