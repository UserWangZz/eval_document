import os
import sys

from rapidfuzz.distance import Levenshtein


class Page:
    def __init__(self, image, image_path, im_id, bbox, rec_threshold, label_base_path):
        self.image = image
        self.image_id = im_id
        self.image_path = image_path
        self.label_base_path = label_base_path
        self.rec_threshold = rec_threshold
        self.bbox = self.bbox2LayoutBbox(bbox)

    def bbox2LayoutBbox(self, bbox):
        result = []
        for box in bbox:
            temp =  LayoutBbox(box['category_id'], box['bbox'], box['score'], box['category_name'], box['layout_crop_image'], box['rec_text_res'])
            result.append(temp)
        return result

    def get_file_context(self):
        # 获取文件路径
        self.ann_page = AnnPage()
        ann = ['标题', '正文', '作者']
        self.file_name = os.path.basename(self.image_path).split('.')[0]
        label_path = os.path.join(self.label_base_path, self.file_name + '.txt')
        with open(label_path, 'r') as f:
            lines = f.readlines()
            i = 0
            while i< len(lines):
                line = lines[i]
                if line.startswith('标题'):
                    i += 1
                    self.ann_page.set_title(lines[i])
                    i += 1
                if line.startswith('正文'):
                    i += 1
                    while i < len(lines) and (not lines[i].startswith('作者') or not lines[i].startswith('标题')):
                        if lines[i] == '\n':
                            i += 1
                            continue
                        self.ann_page.set_content(lines[i])
                        i += 1
                if line.startswith('作者'):
                    i += 1
                    self.ann_page.set_author(lines[i])
                    i += 1

    def recovery_context_from_layoutbbox(self):
        title = ''
        content = ''
        author = ''
        self.rec_result_page = AnnPage()
        for layoutBox in self.bbox:
            if layoutBox.category_name == 'title':
                for rec_result in layoutBox.rec_text_res:
                    rec_content, rec_score = rec_result['rec_result'].split('\t')
                    if float(rec_score) < self.rec_threshold:
                        continue
                    title += rec_content
                title.replace('\n', '')
            elif layoutBox.category_name == 'text':
                for rec_result in layoutBox.rec_text_res:
                    rec_content, rec_score = rec_result['rec_result'].split('\t')
                    if float(rec_score) < self.rec_threshold:
                        continue
                    content += rec_content
                    content.replace('\n', '')
                content += '\n'             # 每一块结束后
            elif layoutBox.category_name == 'anthor':
                for rec_result in layoutBox.rec_text_res:
                    rec_content, rec_score = rec_result['rec_result'].split('\t')
                    if float(rec_score) < self.rec_threshold:
                        continue
                    author += rec_content
                author.replace('\n', '')
        self.rec_result_page.set_title(title)
        self.rec_result_page.set_content(content)
        self.rec_result_page.set_author(author)
    
    def save_page_rec_result(self, save_path):
        save_path = os.path.join(save_path, 'page_rec_content/')
        self.recovery_context_from_layoutbbox()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path+self.file_name+'.txt', 'w') as f:
            length = self.rec_result_page.get_min_list_length()
            for i in range(length):
                f.write("标题：\n"+self.rec_result_page.title[i]+'\n')
                f.write("作者：\n"+self.rec_result_page.author[i]+'\n')
                f.write("正文：\n"+self.rec_result_page.content[i]+'\n')
                f.write('\n')
            if length < len(self.rec_result_page.title):
                f.write("标题：\n"+self.rec_result_page.title[length:]+'\n')
                f.write('\n')
            if length < len(self.rec_result_page.author):
                f.write("作者：\n"+self.rec_result_page.author[length:]+'\n')
                f.write('\n')
            if length < len(self.rec_result_page.content):
                f.write("正文：\n"+self.rec_result_page.content[length:]+'\n')
                f.write('\n')
            f.write('与标注的归一化编辑距离：{}\n'.format(self.cal_edit_distence()))

        

    def cal_edit_distence(self):
        # 从 ann_page 中获取正文
        ann_content = self.ann_page.content

        rec_content = self.rec_result_page.content

        # 将正文拼接为一个字符串
        ann_content = ''.join(ann_content).replace('\n', '')
        rec_content = ''.join(rec_content).replace('\n', '')

        # 计算编辑距离
        # 计算下面代码执行时间
        import time
        start_time = time.time()
        edit_distence = Levenshtein.distance(ann_content, rec_content)
        end_time = time.time()

        print('编辑距离计算时间：', end_time - start_time)

        # print(ann_content)
        # print(rec_content)
        total_num = len(ann_content)
        print(1- edit_distence / (total_num + 1e-10))

        self.edit_dis = 1- edit_distence / (total_num + 1e-10)

        return 1- edit_distence / (total_num + 1e-10)




































        
            
class AnnPage:
    def __init__(self, title=None, content=None, author=None):
        if title is None:
            self.title = []
        else:
            self.title = title
        if content is None:
            self.content = []
        else:
            self.content = content
        if author is None:
            self.author = []
        else:
            self.author = author
    
    def set_title(self, title):
        self.title.append(title)
    
    def set_content(self, content):
        self.content.append(content)
    
    def set_author(self, author):
        self.author.append(author)
    
    def get_total_text_num(self):
        total_num = 0
        for t in self.title:
            total_num += len(t)
        for c in self.content:
            total_num += len(c)
        for a in self.author:
            total_num += len(a)
        return total_num

    def get_min_list_length(self):
        len_title = len(self.title)
        len_content = len(self.content)
        len_author = len(self.author)
        return min(len_title, len_content, len_author)

class LayoutBbox:
    def __init__(self, category_id, box_point, box_score, category_name, crop_image, rec_text_res):
        self.category_id = category_id
        self.box_point = box_point
        self.box_score = box_score
        self.category_name = category_name
        self.crop_image = crop_image
        self.rec_text_res = rec_text_res

    
