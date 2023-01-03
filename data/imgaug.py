
import random
import cv2
import numpy as np
from augraphy import *
from PIL import Image
from torchvision.transforms.functional import resized_crop, crop
from torchvision.transforms import RandomResizedCrop, RandomCrop
from torchvision.transforms import InterpolationMode

def rescale(img, bboxes, target_size=2240):
    h, w = img.shape[0:2]
    scale = target_size / max(h,w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    bboxes = bboxes * scale
    return img, bboxes



def random_resize_crop_synth(augment_targets, size):
    # --------------------------------------------------------------------------------------------------------------#
    image, region_score, affinity_score, confidence_mask = augment_targets

    image = Image.fromarray(image)
    region_score = Image.fromarray(region_score)
    affinity_score = Image.fromarray(affinity_score)
    confidence_mask = Image.fromarray(confidence_mask)

    short_side = min(image.size)
    i,j,h,w = RandomCrop.get_params(image, output_size=(short_side,short_side))

    image = resized_crop(image, i, j, h, w, size=(size, size),
                         interpolation=InterpolationMode.BICUBIC)
    region_score = resized_crop(region_score, i, j, h, w, (size, size),
                                interpolation=InterpolationMode.BICUBIC)
    affinity_score = resized_crop(affinity_score, i, j, h, w, (size, size),
                                  interpolation=InterpolationMode.BICUBIC)
    confidence_mask = resized_crop(confidence_mask, i, j, h, w, (size, size),
                                   interpolation=InterpolationMode.NEAREST)

    image = np.array(image)
    region_score = np.array(region_score)
    affinity_score = np.array(affinity_score)
    confidence_mask = np.array(confidence_mask)
    augment_targets = [image, region_score, affinity_score, confidence_mask]
    # --------------------------------------------------------------------------------------------------------------#

    return augment_targets


def random_resize_crop_ai(augment_targets, scale, ratio, size):
    # --------------------------------------------------------------------------------------------------------------#
    image, region_score, affinity_score, confidence_mask = augment_targets

    image = Image.fromarray(image)
    region_score = Image.fromarray(region_score)
    affinity_score = Image.fromarray(affinity_score)
    confidence_mask = Image.fromarray(confidence_mask)


    i, j, h, w = RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)

    image = resized_crop(image, i, j, h, w, size=(size, size),
                         interpolation=InterpolationMode.BICUBIC)
    region_score = resized_crop(region_score, i, j, h, w, (size, size),
                                interpolation=InterpolationMode.BICUBIC)
    affinity_score = resized_crop(affinity_score, i, j, h, w, (size, size),
                                  interpolation=InterpolationMode.BICUBIC)
    confidence_mask = resized_crop(confidence_mask, i, j, h, w, (size, size),
                                   interpolation=InterpolationMode.NEAREST)
    
    image = np.array(image)
    region_score = np.array(region_score)
    affinity_score = np.array(affinity_score)
    confidence_mask = np.array(confidence_mask)
    augment_targets = [image, region_score, affinity_score, confidence_mask]
    # --------------------------------------------------------------------------------------------------------------#

    return augment_targets




def random_resize_crop(augment_targets, scale, ratio, size, threshold, pre_crop_area=None):
    # --------------------------------------------------------------------------------------------------------------#
    image, region_score, affinity_score, confidence_mask = augment_targets

    image = Image.fromarray(image)
    region_score = Image.fromarray(region_score)
    affinity_score = Image.fromarray(affinity_score)
    confidence_mask = Image.fromarray(confidence_mask)

    #------------------------------#
    if pre_crop_area != None :
        i, j, h, w = pre_crop_area

    else:
        if random.random() < threshold:
            i, j, h, w = RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
        else:
            i, j, h, w = RandomResizedCrop.get_params(image, scale=(1.0, 1.0), ratio=(1.0, 1.0))



    image = resized_crop(image, i, j, h, w, size=(size, size),
                         interpolation=InterpolationMode.BICUBIC)
    region_score = resized_crop(region_score, i, j, h, w, (size, size),
                                interpolation=InterpolationMode.BICUBIC)
    affinity_score = resized_crop(affinity_score, i, j, h, w, (size, size),
                                  interpolation=InterpolationMode.BICUBIC)
    confidence_mask = resized_crop(confidence_mask, i, j, h, w, (size, size),
                                   interpolation=InterpolationMode.NEAREST)


    image = np.array(image)
    region_score = np.array(region_score)
    affinity_score = np.array(affinity_score)
    confidence_mask = np.array(confidence_mask)
    augment_targets = [image, region_score, affinity_score, confidence_mask]
    # --------------------------------------------------------------------------------------------------------------#

    return augment_targets


def random_crop(augment_targets, size):
    # --------------------------------------------------------------------------------------------------------------#
    image, region_score, affinity_score, confidence_mask = augment_targets

    image = Image.fromarray(image)
    region_score = Image.fromarray(region_score)
    affinity_score = Image.fromarray(affinity_score)
    confidence_mask = Image.fromarray(confidence_mask)


    i,j,h,w = RandomCrop.get_params(image, output_size=(size,size))

    image = crop(image, i, j, h, w)
    region_score = crop(region_score, i, j, h, w)
    affinity_score = crop(affinity_score, i, j, h, w)
    confidence_mask = crop(confidence_mask, i, j, h, w)

    image = np.array(image)
    region_score = np.array(region_score)
    affinity_score = np.array(affinity_score)
    confidence_mask = np.array(confidence_mask)
    augment_targets = [image, region_score, affinity_score, confidence_mask]
    # --------------------------------------------------------------------------------------------------------------#

    return augment_targets





def random_crop_with_bbox(augment_targets, word_level_char_bbox, output_size):
    # h, w = augment_targets[0].shape[0:2]
    # th, tw = output_size, output_size
    # crop_h, crop_w = output_size, output_size
    # if w == tw and h == th:
    #     return augment_targets

    # word_bboxes = []
    # if len(word_level_char_bbox) > 0:
    #     for bboxes in word_level_char_bbox:
    #          word_bboxes.append(
    #             [[bboxes[:, :, 0].min(), bboxes[:, :, 1].min()], [bboxes[:, :, 0].max(), bboxes[:, :, 1].max()]])
    # word_bboxes = np.array(word_bboxes, np.int32)

    # if random.random() > 0.6 and len(word_bboxes) > 0:
    #     sample_bboxes = word_bboxes[random.randint(0, len(word_bboxes) - 1)]

    #     left = max(sample_bboxes[1, 0] - output_size, 0)
    #     top = max(sample_bboxes[1, 1] - output_size,0)

    #     if min(sample_bboxes[0, 1], h - th) < top or min(sample_bboxes[0, 0], w - tw) < left:
    #         i = random.randint(0, h - th)
    #         j = random.randint(0, w - tw)
    #     else:
    #         i = random.randint(top, min(sample_bboxes[0, 1], h - th))
    #         j = random.randint(left, min(sample_bboxes[0, 0], w - tw))

    #     crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] - i else th
    #     crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] - j else tw
    # else:
    #     ### train for IC15 dataset####
    #     i = random.randint(0, h - th)
    #     j = random.randint(0, w - tw)

    #     #### train for MLT dataset ###
    #     # i, j = 0, 0
    #     # crop_h, crop_w = h + 1, w + 1  # make the crop_h, crop_w > tw, th

    # for idx in range(len(augment_targets)):
    #     # crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] else th
    #     # crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] else tw

    #     if len(augment_targets[idx].shape) == 3:
    #         augment_targets[idx] = augment_targets[idx][i:i + crop_h, j:j + crop_w, :]
    #     else:
    #         augment_targets[idx] = augment_targets[idx][i:i + crop_h, j:j + crop_w]

    #     if crop_w > tw or crop_h > th:
    #         augment_targets[idx] = padding_image(augment_targets[idx], tw)


    # return augment_targets
    pass

def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs

def random_scale(images, word_level_char_bbox, scale_range):
    scale = random.sample(scale_range, 1)[0]

    for i in range(len(images)):
        images[i] = cv2.resize(images[i], dsize=None, fx=scale, fy=scale)

    for i in range(len(word_level_char_bbox)):
        word_level_char_bbox[i] *= scale

    return images

def random_rotate(images, max_angle):
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(images)):
        img = images[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        if i == len(images) - 1:
            img_rotation = cv2.warpAffine(img, M=rotation_matrix, dsize=(h, w), flags=cv2.INTER_NEAREST)
        else:
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        images[i] = img_rotation
    return images



def random_augraphy(image):
    ink_phase = [

        InkBleed(
            intensity_range=(0.1, 0.2),
            color_range=(0, 16),
            kernel_size=random.choice([(7, 7), (5, 5), (3, 3)]),
            severity=(0.4, 0.6),
            p=0.33,
        ),

        # Letterpress(
        #     n_samples=(100, 400),
        #     n_clusters=(200, 400),
        #     std_range=(500, 3000),
        #     value_range=(150, 224),
        #     value_threshold_range=(96, 128),
        #     blur=1,
        #     p=0.33,
        # ),
        OneOf(
            [
                LowInkRandomLines(
                    count_range=(5, 10),
                    use_consistent_lines=random.choice([True, False]),
                    noise_probability=0.1,
                ),
                LowInkPeriodicLines(
                    count_range=(2, 5),
                    period_range=(16, 32),
                    use_consistent_lines=random.choice([True, False]),
                    noise_probability=0.1,
                ),
            ],
        ),
    ]

    paper_phase = [
        PaperFactory(p=0.33),
        ColorPaper(
            hue_range=(0, 255),
            saturation_range=(10, 40),
            p=0.33,
        ),
        WaterMark(
            watermark_word="random",
            watermark_font_size=(10, 15),
            watermark_font_thickness=(20, 25),
            watermark_rotation=(0, 360),
            watermark_location="random",
            watermark_color="random",
            watermark_method="darken",
            p=0.33,
        ),
        OneOf(
            [
                AugmentationSequence(
                    [
                        NoiseTexturize(
                            sigma_range=(3, 10),
                            turbulence_range=(2, 5),
                        ),
                        BrightnessTexturize(
                            texturize_range=(0.9, 0.99),
                            deviation=0.03,
                        ),
                    ],
                ),
                AugmentationSequence(
                    [
                        BrightnessTexturize(
                            texturize_range=(0.9, 0.99),
                            deviation=0.03,
                        ),
                        NoiseTexturize(
                            sigma_range=(3, 10),
                            turbulence_range=(2, 5),
                        ),
                    ],
                ),
            ],
            p=0.33,
        ),
        Brightness(
            brightness_range=(0.9, 1.1),
            min_brightness=0,
            min_brightness_value=(120, 150),
            p=0.1,
        ),
    ]

    post_phase = [

        OneOf(
            [
                LightingGradient(
                    light_position=None,
                    direction=None,
                    max_brightness=255,
                    min_brightness=0,
                    mode="gaussian",
                    linear_decay_rate=None,
                    transparency=None,
                ),
                Brightness(
                    brightness_range=(0.9, 1.1),
                    min_brightness=0,
                    min_brightness_value=(120, 150),
                ),
            ],
            p=0.33,
        ),
        DirtyDrum(
            line_width_range=(1, 6),
            line_concentration=random.uniform(0.05, 0.15),
            direction=random.randint(0, 2),
            noise_intensity=random.uniform(0.6, 0.95),
            noise_value=(64, 224),
            ksize=random.choice([(3, 3), (5, 5), (7, 7)]),
            sigmaX=0,
            p=0.33,
        ),
        SubtleNoise(
            subtle_range=random.randint(5, 10),
            p=0.33,
        ),
        Jpeg(
            quality_range=(25, 95),
            p=0.33,
        ),

        Markup(
            num_lines_range=(2, 7),
            markup_length_range=(0.5, 1),
            markup_thickness_range=(1, 2),
            markup_type=random.choice(["strikethrough", "crossed", "highlight", "underline"]),
            markup_color="random",
            single_word_mode=False,
            repetitions=1,
            p=0.33,
        ),
        PencilScribbles(
            size_range=(100, 800),
            count_range=(1, 6),
            stroke_count_range=(1, 2),
            thickness_range=(2, 6),
            brightness_change=random.randint(64, 224),
            p=0.33,
        ),
        # BadPhotoCopy(
        #     mask=None,
        #     noise_type=-1,
        #     noise_side="random",
        #     noise_iteration=(1, 2),
        #     noise_size=(1, 3),
        #     noise_value=(128, 196),
        #     noise_sparsity=(0.3, 0.6),
        #     noise_concentration=(0.1, 0.6),
        #     blur_noise=random.choice([True, False]),
        #     blur_noise_kernel=random.choice([(3, 3), (5, 5), (7, 7)]),
        #     wave_pattern=random.choice([True, False]),
        #     edge_effect=random.choice([True, False]),
        #     p=0.33,
        # ),
        Gamma(
            gamma_range=(0.9, 1.1),
            p=0.33,
        ),
        BindingsAndFasteners(
            overlay_types="darken",
            foreground=None,
            effect_type="random",
            ntimes=(2, 6),
            nscales=(0.9, 1.0),
            edge="random",
            edge_offset=(10, 50),
            use_figshare_library=0,
            p=0.33,
        ),



    ]


    pipeline = AugraphyPipeline(ink_phase,paper_phase,post_phase)
    data = pipeline.augment(image)
    augmented = data["output"]

    if len(augmented.shape) < 3:
        augmented = np.stack([augmented, augmented, augmented], axis=-1)

    return augmented