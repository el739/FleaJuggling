import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QWidget, QPushButton, QLabel, QFileDialog, QScrollArea)
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint


class ImageLabel(QLabel):
    """自定义的QLabel，用于显示图片并捕获鼠标位置"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 1px solid gray;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("点击'打开图片'按钮加载图片")
        
        # 启用鼠标跟踪
        self.setMouseTracking(True)
        
        # 存储原始图片
        self.original_pixmap = None
        self.scaled_pixmap = None
        
    def set_image(self, pixmap):
        """设置要显示的图片"""
        self.original_pixmap = pixmap
        self.update_display()
        
    def update_display(self):
        """更新图片显示"""
        if self.original_pixmap:
            # 按比例缩放图片以适应标签大小
            self.scaled_pixmap = self.original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(self.scaled_pixmap)
            
    def resizeEvent(self, event):
        """窗口大小改变时重新缩放图片"""
        super().resizeEvent(event)
        if self.original_pixmap:
            self.update_display()
            
    def mouseMoveEvent(self, event):
        """鼠标移动事件处理"""
        if self.original_pixmap and self.scaled_pixmap:
            # 获取鼠标在标签中的位置
            mouse_pos = event.pos()
            
            # 获取显示图片的实际区域
            pixmap_rect = self.scaled_pixmap.rect()
            label_rect = self.rect()
            
            # 计算图片在标签中的位置（居中显示）
            x_offset = (label_rect.width() - pixmap_rect.width()) // 2
            y_offset = (label_rect.height() - pixmap_rect.height()) // 2
            
            # 检查鼠标是否在图片区域内
            if (x_offset <= mouse_pos.x() <= x_offset + pixmap_rect.width() and
                y_offset <= mouse_pos.y() <= y_offset + pixmap_rect.height()):
                
                # 计算鼠标在显示图片中的相对位置
                relative_x = mouse_pos.x() - x_offset
                relative_y = mouse_pos.y() - y_offset
                
                # 计算缩放比例
                scale_x = self.original_pixmap.width() / pixmap_rect.width()
                scale_y = self.original_pixmap.height() / pixmap_rect.height()
                
                # 计算在原始图片中的像素坐标
                original_x = int(relative_x * scale_x)
                original_y = int(relative_y * scale_y)
                
                # 计算相对坐标（0-1之间）
                relative_coord_x = original_x / self.original_pixmap.width()
                relative_coord_y = original_y / self.original_pixmap.height()
                
                # 更新父窗口的坐标显示
                if self.parent_window:
                    self.parent_window.update_coordinates(
                        original_x, original_y, relative_coord_x, relative_coord_y)
            else:
                # 鼠标不在图片区域内
                if self.parent_window:
                    self.parent_window.clear_coordinates()


class ImageAnalyzer(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("图片物体位置分析工具")
        self.setGeometry(100, 100, 800, 600)
        
        # 创建中心窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建控制面板
        control_layout = QHBoxLayout()
        
        # 打开图片按钮
        self.open_button = QPushButton("打开图片")
        self.open_button.clicked.connect(self.open_image)
        control_layout.addWidget(self.open_button)
        
        # 图片信息显示
        self.image_info_label = QLabel("图片信息：未加载图片")
        control_layout.addWidget(self.image_info_label)
        
        control_layout.addStretch()  # 添加弹性空间
        main_layout.addLayout(control_layout)
        
        # 创建坐标信息面板
        coord_layout = QHBoxLayout()
        
        self.pixel_coord_label = QLabel("像素坐标：(-, -)")
        coord_layout.addWidget(self.pixel_coord_label)
        
        self.relative_coord_label = QLabel("相对坐标：(-.---, -.---)")
        coord_layout.addWidget(self.relative_coord_label)
        
        coord_layout.addStretch()
        main_layout.addLayout(coord_layout)
        
        # 创建滚动区域来容纳图片
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 创建图片显示标签
        self.image_label = ImageLabel(self)
        scroll_area.setWidget(self.image_label)
        
        main_layout.addWidget(scroll_area)
        
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                padding: 8px 16px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLabel {
                font-size: 12px;
                color: #333;
            }
        """)
        
    def open_image(self):
        """打开图片文件对话框"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择图片文件", 
            "", 
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;所有文件 (*)"
        )
        
        if file_path:
            self.load_image(file_path)
            
    def load_image(self, file_path):
        """加载并显示图片"""
        pixmap = QPixmap(file_path)
        
        if pixmap.isNull():
            self.image_info_label.setText("图片信息：加载失败")
            return
            
        # 显示图片
        self.image_label.set_image(pixmap)
        
        # 更新图片信息
        width = pixmap.width()
        height = pixmap.height()
        file_name = os.path.basename(file_path)
        self.image_info_label.setText(
            f"图片信息：{file_name} - {width} x {height} 像素")
        
        # 清空坐标显示
        self.clear_coordinates()
        
    def update_coordinates(self, pixel_x, pixel_y, relative_x, relative_y):
        """更新坐标显示"""
        self.pixel_coord_label.setText(f"像素坐标：({pixel_x}, {pixel_y})")
        self.relative_coord_label.setText(f"相对坐标：({relative_x:.3f}, {relative_y:.3f})")
        
    def clear_coordinates(self):
        """清空坐标显示"""
        self.pixel_coord_label.setText("像素坐标：(-, -)")
        self.relative_coord_label.setText("相对坐标：(-.---, -.---)")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("图片物体位置分析工具")
    app.setApplicationVersion("1.0")
    
    # 创建并显示主窗口
    window = ImageAnalyzer()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()