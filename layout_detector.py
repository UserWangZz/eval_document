from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy


import sys
import ast
import os
import typing
import glob

sys.path.append('{path of PaddleDetection}')

import argparse

from ppdet.core.workspace import load_config, create
from ppdet.utils.cli import merge_args
from ppdet.engine import Trainer
from ppdet.utils.logger import setup_logger
from ppdet.data.source.category import get_categories
from ppdet.utils.visualizer import save_result

from utils.json_result import get_det_res, get_det_poly_res
from PIL import Image, ImageOps, ImageFile

logger = setup_logger('eval')

from tqdm import tqdm





def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def get_infer_results(outs, catid, bias=0):
    """
    Get result at the stage of inference.
    The output format is dictionary containing bbox or mask result.

    For example, bbox result is a list and each element contains
    image_id, category_id, bbox and score.
    """
    if outs is None or len(outs) == 0:
        raise ValueError(
            'The number of valid detection result if zero. Please use reasonable model and check input data.'
        )

    im_id = outs['im_id']

    infer_res = {}
    if 'bbox' in outs:
        if len(outs['bbox']) > 0 and len(outs['bbox'][0]) > 6:
            infer_res['bbox'] = get_det_poly_res(
                outs['bbox'], outs['bbox_num'], im_id, catid, bias=bias)
        else:
            infer_res['bbox'] = get_det_res(
                outs['bbox'], outs['bbox_num'], im_id, catid, bias=bias)

    return infer_res



class ppdet_trainer(Trainer):
    def __init__(self, cfg, mode):
        super(ppdet_trainer, self).__init__(cfg, mode)
    
    def predict_for_doc(self, images, filer_threshold):
        self.dataset.set_images(images)
        loader = create('TestReader')(self.dataset, 0)

        
        imid2path = self.dataset.get_imid2path()

        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(
            self.cfg.metric, anno_file=anno_file
        )
        self.catid2name = catid2name
        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('TestReader')(self.dataset, 0)
            self._flops(flops_loader)
        results = []
        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            # forward


            outs = self.model(data)

            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()
            results.append(outs)

        final_results = []
        for outs in results:
            batch_res = get_infer_results(outs, clsid2catid)
            bbox_num = outs['bbox_num']

            start = 0
            for i, im_id in enumerate(outs['im_id']):
                
                end = start + bbox_num[i]
                bbox_res = batch_res['bbox'][start:end] \
                        if 'bbox' in batch_res else None
                
                new_bbox_res = []
                for box in bbox_res:
                    if box['score'] < filer_threshold:
                        continue
                    box['category_name'] = catid2name[box['category_id']]
                    new_bbox_res.append(box)
                image_path = imid2path[int(im_id)]
                image = Image.open(image_path).convert('RGB')
                image = ImageOps.exif_transpose(image)
                final_results.append(
                    {
                        'im_id': im_id,
                        'image_shape': [image.height, image.width],
                        'image': image,
                        'image_path': image_path,
                        'bbox': new_bbox_res,
                    }
                )
                start = end

        
        return final_results

    def _get_save_image_name(self, output_dir, image_path):
        """
        Get save image name from source image path.
        """
        image_name = os.path.split(image_path)[-1]
        name, ext = os.path.splitext(image_name)
        return os.path.join(output_dir, "{}".format(name)) + ext



class layout_detector():
    def __init__(self, args=None):
        
        self.config = args.layout_config
        self.model_path = args.layout_model_path
        self.infer_dir = args.infer_dir
        self.infer_img = args.infer_img
        self.output_dir = args.output_dir
        self.draw_threshold = args.threshold
        self.visualize = args.visualize
        self.save_results = args.save_results
        
        cfg = load_config(self.config)
        self.cfg = merge_args(cfg, args)

    def run(self):
        self.trainer = ppdet_trainer(self.cfg, mode='test')
        self.trainer.load_weights(self.model_path)
        images = get_test_images(self.infer_dir, self.infer_img)

        # inference
        result = self.trainer.predict_for_doc(images=images, filer_threshold=self.draw_threshold)
        
        return result



if __name__ == '__main__':
    layout_detector = layout_detector()
    result = layout_detector.run()
    print(result)


