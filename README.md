# TABLE CELL SCORE

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

## Description

This package give a metric to evaluate table cell detection based on table cell bounding boxes.

## Installation

You can install the package using pip:

```bash
pip install tabcell_score
```

## Usage

**Input format:** We follow the format of the json label file from [labelme](https://pypi.org/project/labelme/).

```python
{
    "shapes":[
        "label": "cell",
        "points": [
            [
                0.,
                0.,
            ],
            [
                13.,
                13.
            ]
        ]
    ]
}
```

**In code use:** 

```python
from tabcell_score import TabCellScore

# Specify the label and image path
gt_path = r"sample\gt.json"
pred_path = r"sample\pred.json"
img_path = r"sample\image.jpg"

tabcellscore = TabCellScore(iou_threshold = 0.95)

# Output score only
score = tabcellscore(gt_path, pred_path) 

# Output score and visualize on image and save to "result.jpg"
score = tabcellscore(gt_path, pred_path, img_path, "result.jpg") 

# Output score and visualize on empty image and save to "result.jpg"
score = tabcellscore(gt_path, pred_path, None, "result.jpg")

print(score)
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

"Hope this package could help you in table cell evaluation." Phạm Phú Ngọc Trai (from [akaOCR](https://app.akaocr.io/)).