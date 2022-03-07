"""This is the repo which contains the original code to the WACV 2021 paper
"Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows"
by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.
For further information contact Marco Rudolph (rudolph@tnt.uni-hannover.de)"""


import config as c
from train import train
from torch.utils.data import DataLoader

import sys

sys.path.append("/workspaces/ood/")

from ood.data import EmbDataset, EmbDatasetOOD

model_name = c.model_name
emb_size = c.emb_size

# in_distr_train_path = f"../data/predictions/{model_name}_cifar_train.npy"
# in_distr_test_path = f"../data/predictions/{model_name}_cifar_test.npy"
# out_distr_test_path = f"../data/predictions/{model_name}_svhn_test.npy"

# in_distr_train = EmbDataset(in_distr_train_path, emb_size=emb_size)
# in_and_out_distr_test = EmbDatasetOOD(in_distr_test_path, out_distr_test_path, emb_size)


# in_distr_train_dataloader = DataLoader(in_distr_train, batch_size=256, shuffle=False)
# in_and_out_distr_test_dataloader = DataLoader(
#     in_and_out_distr_test, batch_size=256, shuffle=False
# )



model = train(in_distr_train_dataloader, in_and_out_distr_test_dataloader)
