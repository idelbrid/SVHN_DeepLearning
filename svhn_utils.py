import PIL
import numpy as np
# import h5py


def evenly_pad(image, shape):
    height_lacking = shape[1] - image.size[1]
    width_lacking = shape[0] - image.size[0]
    left_pad = width_lacking // 2
    right_pad = width_lacking - left_pad

    top_pad = height_lacking // 2
    bot_pad = height_lacking - top_pad

    return image.crop([-left_pad, -top_pad, image.size[0] + right_pad, image.size[1] + bot_pad])


def file2tensor(filename, shape=(128, 64)):
    with PIL.Image.open(filename) as im:
        width = im.size[0]
        height = im.size[1]
        if im.size[1] < width // 2:
            cropped = evenly_pad(im, [width, width // 2])
        else:
            cropped = evenly_pad(im, [height * 2, height])

        if cropped.size[0] > shape[0]:
            resized = cropped.resize(shape)
        else:
            resized = evenly_pad(cropped, shape)
        return np.asarray(resized)


def parse_digit_struct(digit_struct):
    def maybe_deref(ref):
        if isinstance(ref, float):
            return ref
        else:
            return digit_struct[ref][0,0]

    def parse_bboxes(bboxes_obj):
        bboxes = []
        for href, lref, leref, tref, wref in zip(bboxes_obj['height'], bboxes_obj['label'], bboxes_obj['left'], bboxes_obj['top'], bboxes_obj['width']):
            bbox = dict()
            bbox['height'] = maybe_deref(href[0])
            bbox['width'] = maybe_deref(wref[0])
            bbox['length'] = maybe_deref(leref[0])
            bbox['top'] = maybe_deref(tref[0])
            bbox['label'] = maybe_deref(lref[0])
            bboxes.append(bbox)
        return bboxes

    digitstructdict = dict()
    for bbox_list, name in zip(digit_struct['digitStruct']['bbox'], digit_struct['digitStruct']['name']):
        bboxes_obj = digit_struct[bbox_list[0]]
        names_obj = digit_struct[name[0]]

        name = ''.join(chr(i) for i in names_obj)
        bboxes = parse_bboxes(bboxes_obj)

        digitstructdict[name] = bboxes
    return digitstructdict


def one_hot(dig):
    toreturn = np.zeros([len(dig), 10])
    for i in range(len(dig)):
        toreturn[i, dig[i]-1] = 1
    return toreturn


def pad_one_hots(y_ex):
    toreturn = np.zeros((5, 11))
    toreturn[:len(y_ex), :-1] = y_ex
    toreturn[len(y_ex):, -1] = 1
    return toreturn


def shuffle(x, *args, seed=None):
    idx = np.arange(len(x))
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(idx)
    x = x[idx]
    targs = [arg[idx] for arg in args]
    if len(targs) > 0:
        return [x] + targs
    else:
        return x
