import json
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


class textRecognizor():
    def __init__(self, args=None):
        config = load_config(args.eval_rec_config_path)

        self.config = config
        self.config['Global']['checkpoints'] = args.rec_model_path


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

        self.post_process_class = build_post_process(self.config['PostProcess'],
                                            global_config)
        
        # build model
        if hasattr(self.post_process_class, 'character'):
            char_num = len(getattr(self.post_process_class, 'character'))
            if self.config["Architecture"]["algorithm"] in ["Distillation",
                                                    ]:  # distillation model
                for key in self.config["Architecture"]["Models"]:
                    if self.config["Architecture"]["Models"][key]["Head"][
                            "name"] == 'MultiHead':  # multi head
                        out_channels_list = {}
                        if self.config['PostProcess'][
                                'name'] == 'DistillationSARLabelDecode':
                            char_num = char_num - 2
                        if self.config['PostProcess'][
                                'name'] == 'DistillationNRTRLabelDecode':
                            char_num = char_num - 3
                        out_channels_list['CTCLabelDecode'] = char_num
                        out_channels_list['SARLabelDecode'] = char_num + 2
                        out_channels_list['NRTRLabelDecode'] = char_num + 3
                        self.config['Architecture']['Models'][key]['Head'][
                            'out_channels_list'] = out_channels_list
                    else:
                        self.config["Architecture"]["Models"][key]["Head"][
                            "out_channels"] = char_num
            elif self.config['Architecture']['Head'][
                    'name'] == 'MultiHead':  # multi head
                out_channels_list = {}
                char_num = len(getattr(self.post_process_class, 'character'))
                if self.config['PostProcess']['name'] == 'SARLabelDecode':
                    char_num = char_num - 2
                if self.config['PostProcess']['name'] == 'NRTRLabelDecode':
                    char_num = char_num - 3
                out_channels_list['CTCLabelDecode'] = char_num
                out_channels_list['SARLabelDecode'] = char_num + 2
                out_channels_list['NRTRLabelDecode'] = char_num + 3
                self.config['Architecture']['Head'][
                    'out_channels_list'] = out_channels_list
            else:  # base rec model
                self.config["Architecture"]["Head"]["out_channels"] = char_num
        self.model = build_model(self.config['Architecture'])

        load_model(self.config, self.model)

        transforms = []
        for op in self.config['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                continue
            elif op_name in ['RecResizeImg']:
                op[op_name]['infer_mode'] = True
            elif op_name == 'KeepKeys':
                if self.config['Architecture']['algorithm'] == "SRN":
                    op[op_name]['keep_keys'] = [
                        'image', 'encoder_word_pos', 'gsrm_word_pos',
                        'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'
                    ]
                elif self.config['Architecture']['algorithm'] == "SAR":
                    op[op_name]['keep_keys'] = ['image', 'valid_ratio']
                elif self.config['Architecture']['algorithm'] == "RobustScanner":
                    op[op_name][
                        'keep_keys'] = ['image', 'valid_ratio', 'word_positons']
                else:
                    op[op_name]['keep_keys'] = ['image']
            transforms.append(op)
        global_config['infer_mode'] = True
        
        self.ops = create_operators(transforms, global_config)
        
        self.model.eval()

    
        
    def __call__(self, dict_res):
        block_img = dict_res['layout_crop_image']
        det_boxes = dict_res['text_det_boxes']


        rec_text_res = []
        
        if dict_res['category_name'] == 'title' or dict_res['category_name'] == 'anthor':

            info = self.infer_img(block_img)
            rec_text_res.append({'rec_result':info})
        else:
            for box in det_boxes:
                temp_dict = {}
                crop_rec_img = block_img[box[0][1]:box[3][1], box[0][0]:box[2][0]]
                temp_dict['det_box'] = np.array(box)
                
                info = self.infer_img(crop_rec_img)
                
                temp_dict['rec_result'] = info
                rec_text_res.append(temp_dict)

        dict_res['rec_text_res'] = rec_text_res
        return dict_res
    

    def infer_img(self, img):
        data = {'image': img}

        batch = transform(data, self.ops)

        images = np.expand_dims(batch[0], axis=0)
        images = paddle.to_tensor(images)

        preds = self.model(images)

        post_result = self.post_process_class(preds)

        if isinstance(post_result, dict):
            rec_info = dict()
            for key in post_result:
                if len(post_result[key][0]) >= 2:
                    rec_info[key] = {
                        "label": post_result[key][0][0],
                        "score": float(post_result[key][0][1]),
                    }
            info = json.dumps(rec_info, ensure_ascii=False)
        elif isinstance(post_result, list) and isinstance(post_result[0],
                                                            int):
            # for RFLearning CNT branch 
            info = str(post_result[0])
        else:
            if len(post_result[0]) >= 2:
                info = post_result[0][0] + "\t" + str(post_result[0][1])
        
        return info


if __name__ == '__main__':

    from eval_doc import parse_args
    args = parse_args()

    text_rec = textRecognizor(args)
