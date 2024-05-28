import sys
from typing import Any
sys.path.append('/paddle/PaddleOCR')
import os
import yaml
import paddle
import paddle.distributed as dist
import numpy as np


from ppocr.utils.logging import get_logger
from tools.program import check_device, ArgsParser
from ppocr.utils.utility import print_dict

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return config


class textDetector():
    def __init__(self, args=None):
        config = load_config(args.eval_det_config_path)

        self.config = config
        self.config['Global']['checkpoints'] = args.det_model_path

        # 对det的一些调参
        self.config['PostProcess']['box_thresh'] = 0.3
        self.config['PostProcess']['score_mode'] = 'slow'


        self.logger = get_logger(log_file=None)
        # check if set use_gpu=True in paddlepaddle cpu version
        use_gpu = self.config['Global'].get('use_gpu', False)
        use_xpu = self.config['Global'].get('use_xpu', False)
        use_npu = self.config['Global'].get('use_npu', False)
        use_mlu = self.config['Global'].get('use_mlu', False)

        if use_xpu:
            device = 'xpu:{0}'.format(os.getenv('FLAGS_selected_xpus', 0))
        elif use_npu:
            device = 'npu:{0}'.format(os.getenv('FLAGS_selected_npus', 0))
        elif use_mlu:
            device = 'mlu:{0}'.format(os.getenv('FLAGS_selected_mlus', 0))
        else:
            device = 'gpu:{}'.format(dist.ParallelEnv()
                                    .dev_id) if use_gpu else 'cpu'
        
        check_device(use_gpu, use_xpu, use_npu, use_mlu)

        self.device = paddle.set_device(device)

        self.config['Global']['distributed'] = dist.get_world_size() != 1

        # print_dict(self.config, self.logger)

        self.logger.info('train with paddle {} and device {}'.format(paddle.__version__,
                                                            device))
        

        global_config = self.config['Global']

        self.model = build_model(self.config['Architecture'])

        load_model(self.config, self.model)

        self.post_process_class = build_post_process(self.config['PostProcess'])

        transforms = []
        for op in self.config['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                continue
            elif op_name == 'KeepKeys':
                op[op_name]['keep_keys'] = ['image', 'shape']
            transforms.append(op)

        self.ops = create_operators(transforms, global_config)
        self.model.eval()


    def __call__(self, image):
        
        data = {'image': image}

        batch = transform(data, self.ops)

        images = np.expand_dims(batch[0], axis=0)
        shape_list = np.expand_dims(batch[1], axis=0)
        images = paddle.to_tensor(images)
        preds = self.model(images)
        post_result = self.post_process_class(preds, shape_list)


        return post_result
    




if __name__ == '__main__':

    from eval_doc import parse_args
    args = parse_args()

    text_det = textDetector(args)
