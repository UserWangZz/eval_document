from PIL import Image

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


if __name__ == '__main__':
    bbox = [
            {'image_id': 0, 'category_id': 1, 'bbox': [1277.7960205078125, 1154.9967041015625, 1431.5389404296875, 241.740234375], 'score': 0.7250540852546692, 'category_name': 'title'}, 
            {'image_id': 0, 'category_id': 2, 'bbox': [2744.310302734375, 1169.5223388671875, 382.538818359375, 168.060546875], 'score': 0.6241254210472107, 'category_name': 'anthor'}, 
            {'image_id': 0, 'category_id': 3, 'bbox': [2238.0263671875, 1436.03125, 971.5732421875, 2460.1669921875], 'score': 0.9594253301620483, 'category_name': 'text'}, 
            {'image_id': 0, 'category_id': 3, 'bbox': [1229.297607421875, 1469.37255859375, 967.80126953125, 3929.927734375], 'score': 0.9577386975288391, 'category_name': 'text'}, 
            {'image_id': 0, 'category_id': 3, 'bbox': [248.0453643798828, 2898.317138671875, 952.8116912841797, 2427.250732421875], 'score': 0.9501455426216125, 'category_name': 'text'}
        ]
    img = Image.open('{An image path}')
    width = img.size[0]  # 图片宽度 3240 高度 5760
    res = sorted_layout_boxes(bbox, width)
    print(res)
