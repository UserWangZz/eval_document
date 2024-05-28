import os
import sys
sys.path.append('{path of PaddleOCR}')
sys.path.append('{path of PaddleDetection}')
import argparse
import ast
import cv2
import numpy as np

from layout_detector import layout_detector
from text_detector import textDetector
from text_recognizor import textRecognizor
from visualizer import visualize_results
from tqdm import tqdm
from tools.infer.predict_system import sorted_boxes
from tools.infer.utility import get_rotate_crop_image
from ppdet.utils.logger import setup_logger
from tools.infer_det import draw_det_res

from tools.infer.utility import str2bool

from page import Page, LayoutBbox

logger = setup_logger('eval')


def parse_args():
    parser = argparse.ArgumentParser(description='Eval document')
    parser.add_argument('--infer_dir', type=str, default=None, help='Directory for images to perform inference on.')
    parser.add_argument("--infer_img",type=str,default=None,help="Image path, has higher priority over --infer_dir")
    parser.add_argument("--label_path", type=str, default=None, help='label file path')
    parser.add_argument("--output_dir",type=str,default="output",help="Directory for storing the output visualization files.")
    parser.add_argument("--rec_filter_threshold",type=float,default=0.5,help="Threshold to filter the rec result.")


    # layout detection
    parser.add_argument('--layout_config', type=str, default=None, help='config file path')
    parser.add_argument('--layout_model_path', type=str, default=None, help='layout checkpoint file path')
    parser.add_argument("--threshold",type=float,default=0.5,help="Threshold to filter the result.")
    parser.add_argument("--save_results",type=bool,default=False,help="Whether to save inference results to output_dir.")
    parser.add_argument("--visualize",type=ast.literal_eval,default=True,help="Whether to save visualize results to output_dir.")

    # text detection
    parser.add_argument("--eval_det_config_path", type=str, help="det config path")
    parser.add_argument("--det_model_path", type=str, help="det model path")


    # text recognition
    parser.add_argument("--eval_rec_config_path", type=str, help="rec config path")
    parser.add_argument("--rec_model_path", type=str, help="rec model path")

    parser = parser.parse_args()
    return parser


def sorted_layout_boxes(res, w):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        res(list):ppstructure results
        w(int):image width
    return:
        sorted results(list)
    """
    num_boxes = len(res)
    if num_boxes == 1:
        res[0]['layout'] = 'single'
        return res

    sorted_boxes = sorted(res, key=lambda x: (x['bbox'][0], x['bbox'][1]))
    _boxes = list(sorted_boxes)

    new_res = []
    res_left = []
    res_mid = []
    res_right = []
    i = 0

    while True:
        if i >= num_boxes:
            break
        # if _boxes[i]['category_name'] != 'text':
        #     i+=1
        #     continue
        # 检查是不是三列图片
        if _boxes[i]['bbox'][0] > w / 4 and _boxes[i]['bbox'][0] + _boxes[i]['bbox'][2] < 3 * w / 4:
            _boxes[i]['layout'] = 'double'
            res_mid.append(_boxes[i])
            i += 1
        # 检查bbox是否在左侧
        elif _boxes[i]['bbox'][0] < w / 4 and _boxes[i]['bbox'][0] + _boxes[i]['bbox'][2] < 3 * w / 5:
            _boxes[i]['layout'] = 'double'
            res_left.append(_boxes[i])
            i += 1
        elif _boxes[i]['bbox'][0] > 2 * w / 5 and _boxes[i]['bbox'][0] + _boxes[i]['bbox'][2] < w:
            _boxes[i]['layout'] = 'double'
            res_right.append(_boxes[i])
            i += 1
        else:
            new_res += res_left
            new_res += res_right
            _boxes[i]['layout'] = 'single'
            new_res.append(_boxes[i])
            res_left = []
            res_right = []
            i += 1
    
    res_left = sorted(res_left, key=lambda x: (x['bbox'][1]))
    res_mid = sorted(res_mid, key=lambda x: (x['bbox'][1]))
    res_right = sorted(res_right, key=lambda x: (x['bbox'][1]))
    
    if res_left:
        new_res += res_left
    if res_mid:
        new_res += res_mid
    if res_right:
        new_res += res_right
    
    return new_res


def main():
    total_args = parse_args()

    layout = layout_detector(total_args)

    logger.info("****************LayoutAnalysis****************")

    layout_result = layout.run()

    logger.info('****************End****************')

    # 每页图像排序
    pages = []
    for i, res in enumerate(layout_result):
        res  = sorted_layout_boxes(res['bbox'], w=res['image_shape'][1])
        layout_result[i]['bbox'] = res

        if total_args.visualize:
            vis_image = layout_result[i]['image'].copy()
            vis_image = visualize_results(
                        vis_image, layout_result[i]['bbox'], mask_res=None, segm_res=None, keypoint_res=None,
                        pose3d_res=None, im_id=int(layout_result[i]['im_id']), catid2name=layout.trainer.catid2name, threshold=total_args.threshold)
            save_name = layout.trainer._get_save_image_name(total_args.output_dir,
                                                layout_result[i]['image_path'])
            
            if not os.path.exists(total_args.output_dir):
                os.makedirs(total_args.output_dir)

            logger.info("Detection bbox results save in {}".format(
                save_name))
            vis_image.save(save_name, quality=95)
        
        # 进行crop
        for j, dict_res in enumerate(res):
            bbox = dict_res['bbox']
            # bbox 格式转换
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
            bbox = np.array([[xmin, ymin],[xmax, ymin],[xmax,ymax],[xmin,ymax]], dtype=np.int32)

            # 转为opencv
            img = np.array(layout_result[i]['image'])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # temp_img = get_rotate_crop_image(img, bbox)
            temp_img = img[bbox[0][1]:bbox[2][1], bbox[0][0]:bbox[2][0]].copy()
            cv2.imwrite('./temp/{}_{}.jpg'.format(i,j), temp_img)
            
            # 将截取结果保存
            res[j]['bbox'] = bbox
            res[j]['layout_crop_image'] = temp_img
        
        # layout_result[i]['bbox'] = res

    text_detector = textDetector(total_args)

    text_recognizor = textRecognizor(total_args)

    text_det_result = None
    for i, res in enumerate(layout_result):
        logger.info(layout_result[i]['image_path'])
        logger.info("****************{} Detecting and Recognizing****************".format(layout_result[i]['image_path']))
        for j, dict_res in enumerate(tqdm(res['bbox'])):
            crop_image = dict_res['layout_crop_image']
            if dict_res['category_name'] == 'title' or dict_res['category_name'] == 'anthor':
                # 判断标题和作者是否是竖直方向文本
                h, w, _ = crop_image.shape
                if h/w >= 2:
                    # 逆时针旋转90度
                    # center = (w / 2, h / 2)
                    # M = cv2.getRotationMatrix2D(center, -90, 1.0)
                    # rotated = cv2.warpAffine(crop_image, M, (w, h))
                    # crop_image = rotated
                    crop_image = cv2.rotate(crop_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    cv2.imwrite('./temp/ttt.jpg', crop_image)
                    dict_res['layout_crop_image'] = crop_image
            
            
            text_det_result = text_detector(crop_image)[0]['points']

            # 临时可视化det 阶段
            draw_det_res(text_det_result,config=None, img=crop_image, img_name='{}_{}.jpg'.format(i,j), save_path='/paddle/eval_doc/output_eval/read_2_detshow/')


            # 从crop_image中继续crop文本行
            text_det_result = sorted_boxes(text_det_result)
            dict_res['text_det_boxes'] = text_det_result.copy()
            
            dict_res = text_recognizor(dict_res)
            

            dict_res.pop('text_det_boxes')
        page = Page(image_path=res['image_path'], image=res['image'], im_id=res['im_id'], 
                    bbox=res['bbox'], rec_threshold=total_args.rec_filter_threshold, label_base_path=total_args.label_path)
        pages.append(page)
    logger.info("****************End****************")
    # print(layout_result[0]['bbox'])

    logger.info('Save ocr result to txt file')
    total_edit = 0
    for page in tqdm(pages):
        page.get_file_context()
        page.save_page_rec_result(total_args.output_dir)
        total_edit += page.cal_edit_distence()
    result_edit = total_edit / len(pages)
    logger.info('Average edit distance: {}'.format(result_edit))




if __name__ == '__main__':
    main()
    