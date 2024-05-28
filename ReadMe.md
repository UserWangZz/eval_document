# Recover document Page

This repo dependes on [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

## 1.Pipline and function
### 1.1 Pipeline

Layout Analysis -> Box sort -> TextDetection -> TextRecognition -> Result

### 1.2 Function

This repo based on the Layout Analysis to detection the element in the document page, then crop it and use OCR tecnhnique to get the text, author and title, etc which is based on the Layout model you trained.

Next, for the title and author inorder to get the right result, We not use Text Detection model to Detection the text region, because in some magazine, the title and author are used vertical layout.

Finally, we output the result into a txt file, which conclude page's element.

## 2.Install
### 2.1  Prepare PaddleOCR and PaddleDetection

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR
git clone https://github.com/PaddlePaddle/PaddleDetection
```
Install the PaddleOCR and PaddleDetection environment.

### 2.2  Clone Eval_Document
```bash
git clone https://github.com/UserWangZz/eval_document.git
```

### 2.3  Install the requirements
```bash
pip install -r requirements.txt
```

## 3.Usage
### 3.1  Prepare the model and data

#### 3.1.1  Prepare the Layout Analysis model

You can download the model from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/docs/models_list.md) or train a Layout Analysis model by yourself.

[How to train a Layout Analysis model?](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/layout/README_ch.md)

#### 3.1.2  Prepare the Text Detection and Text Recognition model

You can download the model from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/models_list.md) or train a Text Detection model by yourself.

[Train a Text Detection model](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/detection.md)

[Train a Text Recognition model](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/recognition.md)


### 3.2  Run the code

Here is a reference config to run the code.

eval a picture:
```bash
python eval_doc.py --infer_img={path of your infer img} \
                   --label_path={label path if you have} \
                   --output_dir={path of output dir} \
                   --rec_filter_threshold=0.5 \
                   --layout_config={Layout Analysis model config} \
                   --layout_model_path={path of Layout Analysis model} \
                   --threshold=0.5 \
                   --eval_det_config_path={Text Detection model config path} \
                   --det_model_path={Text Detection model path} \
                   --eval_rec_config_path={Text Recognition model config path} \
                   --rec_model_path={Text Recognition model path}
```

eval a folder:
```bash
python eval_doc.py --infer_dir={path of your infer dir} \
                   --label_path={label path if you have} \
                   --output_dir={path of output dir} \
                   --rec_filter_threshold=0.5 \
                   --layout_config={Layout Analysis model config} \
                   --layout_model_path={path of Layout Analysis model} \
                   --threshold=0.5 \
                   --eval_det_config_path={Text Detection model config path} \
                   --det_model_path={Text Detection model path} \
                   --eval_rec_config_path={Text Recognition model config path} \
                   --rec_model_path={Text Recognition model path}
```
