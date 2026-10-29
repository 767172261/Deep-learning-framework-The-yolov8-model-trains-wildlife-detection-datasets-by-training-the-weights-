深度学习框架yolov8模型训练野生动物检测数据集 通过训练野生动物数据集的权重 如何建立深度学习yolov8野生动物检测系统
@[toc]


野生动物检测数据集，含20806张训练，1978张验证图像，981张测试图像。包含80种类别。YOLO标注，TXT格式
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/538aa8372bf049cfa7eda07db2873ff8.png)

nC: 80
names: ['Bear', 'Brown bear, 'Bull", 'Butterfly', 'Camel', 'Canary', 'Caterpillar', 'Cattle', 'Centipede', 'Cheetah', 'Chicken', 'Crab', 'Crocodile', 'Deer', 'Duck', 'Eagle', 'Elephant', 'Fish', 'Fox', 'Frog', 'Giraffe',
'Goat', 'Goldfish', 'Goose', 'Hamster', 'Harbor- seal', 'Hedgehog', 'Hippopotamus', 'Horse', 'Jaguar;, 'Jellyfish', 'Kangaroo', 'Koala', 'Ladybug', 'Leopard', 'Lion', 'Lizard', 'Lynx', 'Magpie', 'Monkey', 'Moths
and- butterflies';, 'Mouse', "'Mule', 'Ostrich';, 'Otter', 'OwI', 'Panda', 'Parrot', 'Penguin', 'Pig'; 'Polar- bear'; 'Rabbit', 'Raccoon'; 'Raven', 'Red panda', 'Rhinoceros', 'Scorpion', 'Sea lion', 'Sea turtle', 'Seahorse',
'Shark', 'Sheep', 'Shrimp';, 'Snail', 'Snake', 'Sparrow', 'Spider', 'Squid', 'Squirrel', 'Starfish', 'Swan', 'Tick', 'Tiger', 'Tortoise', 'Turkey', 'Turtle';, 'Whale';, 'Woodpecker', "Worm';, 'Zebra']
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6f42dfe33f80484d8b1e488ddadf1119.png)

1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/050c8a7425014242856305a09cefd0fe.png)
\使用YOLOv8训练这个野生动物检测数据集，并基于训练好的权重建立一套检测系统，我们将从环境搭建、数据准备、模型训练到推理与部署.
以下文字及代码仅供参考学习使用

### 环境搭建

#### 1. 安装CUDA驱动
确保你的显卡支持CUDA，并安装相应的CUDA驱动程序。你可以访问NVIDIA官网下载与你的显卡和操作系统匹配的CUDA驱动。

#### 2. 安装Anaconda
从[Anaconda官网](https://www.anaconda.com/products/distribution)下载并安装Anaconda。这是一个开源的Python发行版，它包含了大量科学计算、数据分析和机器学习相关的库。

#### 3. 创建Python虚拟环境
打开Anaconda Prompt（或命令行工具），创建一个新的虚拟环境：
```bash
conda create -n wildlife_detection python=3.9
conda activate wildlife_detection
```

#### 4. 安装必要的依赖项
在激活的虚拟环境中，安装所需的Python包：
```bash
pip install opencv-python-headless PyQt5 ultralytics torch torchvision torchaudio
```
注意：如果需要GPU加速，请确保安装了对应的CUDA版本的PyTorch。

### 数据集准备与预处理

确保你的数据集已按照以下结构组织：
```
wildlife_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```
并且标注文件是YOLO格式的txt文件。

### 创建`data.yaml`配置文件

创建一个`data.yaml`文件，定义数据集路径和类别信息：
```yaml
train: ./wildlife_dataset/images/train
val: ./wildlife_dataset/images/val
test: ./wildlife_dataset/images/test

nc: 80 # 类别数量
names: ['Bear', 'Brown bear', 'Bull', 'Butterfly', 'Camel', 'Canary', 'Caterpillar', 'Cattle', 'Centipede', 'Cheetah', 'Chicken', 'Crab', 'Crocodile', 'Deer', 'Duck', 'Eagle', 'Elephant', 'Fish', 'Fox', 'Frog', 'Giraffe', 'Goat', 'Goldfish', 'Goose', 'Hamster', 'Harbor-seal', 'Hedgehog', 'Hippopotamus', 'Horse', 'Jaguar', 'Jellyfish', 'Kangaroo', 'Koala', 'Ladybug', 'Leopard', 'Lion', 'Lizard', 'Lynx', 'Magpie', 'Monkey', 'Moths and butterflies', 'Mouse', 'Mule', 'Ostrich', 'Otter', 'Owl', 'Panda', 'Parrot', 'Penguin', 'Pig', 'Polar-bear', 'Rabbit', 'Raccoon', 'Raven', 'Red panda', 'Rhinoceros', 'Scorpion', 'Sea lion', 'Sea turtle', 'Seahorse', 'Shark', 'Sheep', 'Shrimp', 'Snail', 'Snake', 'Sparrow', 'Spider', 'Squid', 'Squirrel', 'Starfish', 'Swan', 'Tick', 'Tiger', 'Tortoise', 'Turkey', 'Turtle', 'Whale', 'Woodpecker', 'Worm', 'Zebra']
```

### YOLOv8模型训练

首先，确保你已经安装了Ultralytics YOLOv8：
```bash
pip install ultralytics
```

然后，运行以下命令开始训练模型。你可以根据自己的需求调整参数。
```bash
yolo task=detect mode=train model=yolov8s.yaml data=data.yaml epochs=100 batch=16 imgsz=640
```
- `model=yolov8s.yaml`：指定要使用的YOLOv8模型架构（如`s`代表small）。
- `epochs=100`：设置训练周期数。
- `batch=16`：设置批量大小。
- `imgsz=640`：设置输入图像尺寸。

### 推理与部署

训练完成后，可以加载模型并对新图片或视频进行预测：
```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')  # 替换为你的模型路径

# 对一张图片进行预测
results = model('path_to_your_image.jpg')

# 或者对视频进行预测
results = model('path_to_your_video.mp4')
```

### 性能评估

可以使用验证集来评估模型性能：
```bash
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml
```

### 模型导出

如果你想将模型导出以用于其他平台，可以使用如下命令：
```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

### 建立检测系统

为了基于训练好的权重建立检测系统，我们可以构建一个简单的图形用户界面（GUI）来上传图片或视频并展示检测结果。这里提供了一个使用PyQt5的例子：

```python
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from ultralytics import YOLO

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("野生动物检测系统")
        self.setGeometry(100, 100, 800, 600)
        
        # 初始化模型
        self.model = YOLO('runs/detect/train/weights/best.pt')  # 替换为你的模型路径
        
        # 创建界面组件
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        self.upload_button = QPushButton("上传图片", self)
        self.upload_button.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_button)
        
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)
        
        self.detect_button = QPushButton("开始检测", self)
        self.detect_button.clicked.connect(self.perform_detection)
        layout.addWidget(self.detect_button)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def upload_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '打开图片', '', "Image files (*.jpg *.png)")
        if fname:
            self.current_image_path = fname
            pixmap = QPixmap(fname)
            self.image_label.setPixmap(pixmap)

    def perform_detection(self):
        results = self.model(self.current_image_path)
        image = cv2.imread(self.current_image_path)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f'{int(cls)} {conf:.2f}'
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
```


