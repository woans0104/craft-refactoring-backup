import os
import sys
import psutil
import time
import yaml
import argparse

import torch

from config.load_config import load_yaml, DotDict
from model.craft import CRAFT
from model.craft_resnet import UNetWithResnet50Encoder
from data.dataset import ICDAR2015, SynthTextDataSet, PreScripTion
#from data.dataset_prescrip import PreScripTion
from data.imgproc import denormalizeMeanVariance
from utils.util import saveInput, saveImage, copyStateDict
# from decorator_wraps import memory_printer

if __name__ == "__main__":
    # print(sys.path)

    parser = argparse.ArgumentParser(description="CRAFT IC15 Train")
    parser.add_argument("--yaml",
                        "--yaml_file_name",
                        default="ic15_train",
                        type=str,
                        help="Load configuration")
    args = parser.parse_args()
    config = load_yaml(args.yaml)
    config = DotDict(config)
    res_dir = os.path.join(os.path.join("exp", args.yaml), "{}".format(args.yaml))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    start_time = time.time()
    print(yaml.dump(config))

    #craft = CRAFT(pretrained=True, amp=config.train.amp)
    craft = UNetWithResnet50Encoder(pretrained=False, amp=config.train.amp)


    if config.train.ckpt_path is not None:
        net_param = torch.load(config.train.ckpt_path)
        craft.load_state_dict(copyStateDict(net_param["craft"]))
    craft = craft.cuda()



    ic15_dataset = PreScripTion(
        output_size=config.train.data.output_size,
        data_dir=config.data_dir.prescrip_train,
        mean=config.train.data.mean,
        variance=config.train.data.variance,
        gauss_init_size=config.train.data.gauss_init_size,
        gauss_sigma=config.train.data.gauss_sigma,
        enlarge_region=config.train.data.enlarge_region,
        enlarge_affinity=config.train.data.enlarge_affinity,
        watershed_param=config.train.data.watershed,
        aug=config.train.data.prescrip_aug,
        vis_test_dir=config.vis_test_dir,
        vis_opt=config.train.data.vis_opt,
        pseudo_vis_opt=config.train.data.pseudo_vis_opt,
    )

    #
    # ic15_dataset = SynthTextDataSet(
    #     output_size=config.train.data.output_size,
    #     data_dir=config.data_dir.synthtext,
    #     saved_gt_dir=None,
    #     gauss_init_size=config.train.data.gauss_init_size,
    #     gauss_sigma=config.train.data.gauss_sigma,
    #     enlarge_region=config.train.data.enlarge_region,
    #     enlarge_affinity=config.train.data.enlarge_affinity,
    #     aug=config.train.data.syn_aug,
    #     vis_test_dir=config.vis_test_dir,
    #     vis_opt=config.train.data.vis_opt,
    #     sample=config.train.data.syn_sample
    #
    # )



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ic15_dataset.update_model(craft)
    ic15_dataset.update_device(device)

    ic15_train_loader = torch.utils.data.DataLoader(
        ic15_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )

    batch_ic = iter(ic15_train_loader)
    ic15_images, ic15_region_label, ic15_affi_label, ic15_confidence_mask = next(
        batch_ic
    )
    print(ic15_images.numpy()[0].transpose(1, 2, 0).shape, ic15_images.numpy()[0].transpose(1, 2, 0).dtype, ic15_images.numpy()[0].transpose(1, 2, 0).max())
    print(ic15_region_label.numpy()[0].shape, ic15_region_label.numpy()[0].dtype, ic15_region_label.numpy()[0].max())
    print(ic15_affi_label.numpy()[0].shape, ic15_affi_label.numpy()[0].dtype, ic15_affi_label.numpy()[0].max())
    print(ic15_confidence_mask.numpy()[0].shape, ic15_confidence_mask.numpy()[0].dtype, ic15_confidence_mask.numpy()[0].max())

    if config.train.data.vis_opt:
        for i in range(config.train.batch_size):
            saveInput(
                f'test_img_{i}',
                res_dir,
                denormalizeMeanVariance(ic15_images.numpy()[i].transpose(1, 2, 0)),
                ic15_region_label.numpy()[i],
                ic15_affi_label.numpy()[i],
                ic15_confidence_mask.numpy()[i]
            )

    print(f"elapsed time : {time.time() - start_time}")