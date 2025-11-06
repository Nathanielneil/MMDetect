#!/usr/bin/env python3
"""
手势分类器训练工具
支持多种机器学习算法
"""

import json
import numpy as np
import pickle
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class GestureClassifierTrainer:
    """手势分类器训练器"""

    def __init__(self, data_file):
        """
        初始化

        Args:
            data_file: 数据集JSON文件路径
        """
        self.data_file = data_file
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.classifier = None
        self.label_map = {}

    def load_data(self):
        """加载数据"""
        print(f"加载数据: {self.data_file}")

        with open(self.data_file, 'r') as f:
            dataset = json.load(f)

        print(f"总样本数: {len(dataset)}")

        # 提取特征和标签
        X = []
        y = []

        for sample in dataset:
            X.append(sample['landmarks'])  # 63维特征
            y.append(sample['label'])

        self.X = np.array(X)
        self.y = np.array(y)

        # 创建标签映射
        unique_labels = np.unique(self.y)
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.inverse_label_map = {idx: label for label, idx in self.label_map.items()}

        # 转换为数字标签
        y_numeric = np.array([self.label_map[label] for label in self.y])
        self.y = y_numeric

        print(f"特征维度: {self.X.shape}")
        print(f"类别数: {len(unique_labels)}")
        print(f"类别: {list(unique_labels)}")

        # 统计各类样本数
        from collections import Counter
        label_counts = Counter([self.inverse_label_map[y] for y in self.y])
        print("\n各类样本数:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

    def split_data(self, test_size=0.2, random_state=42):
        """划分训练集和测试集"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y  # 分层采样,保持类别比例
        )

        print(f"\n数据集划分:")
        print(f"  训练集: {len(self.X_train)} 样本")
        print(f"  测试集: {len(self.X_test)} 样本")

    def train(self, algorithm='random_forest', **kwargs):
        """
        训练分类器

        Args:
            algorithm: 算法类型
                - 'random_forest': 随机森林(推荐)
                - 'svm': 支持向量机
                - 'knn': K近邻
                - 'mlp': 多层感知机
                - 'gradient_boosting': 梯度提升
            **kwargs: 算法参数
        """
        print(f"\n使用算法: {algorithm}")

        if algorithm == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=42,
                n_jobs=-1
            )

        elif algorithm == 'svm':
            self.classifier = SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                gamma=kwargs.get('gamma', 'scale'),
                probability=True,  # 启用概率预测
                random_state=42
            )

        elif algorithm == 'knn':
            self.classifier = KNeighborsClassifier(
                n_neighbors=kwargs.get('n_neighbors', 5),
                weights=kwargs.get('weights', 'distance'),
                n_jobs=-1
            )

        elif algorithm == 'mlp':
            self.classifier = MLPClassifier(
                hidden_layer_sizes=kwargs.get('hidden_layers', (128, 64)),
                activation=kwargs.get('activation', 'relu'),
                max_iter=kwargs.get('max_iter', 500),
                random_state=42
            )

        elif algorithm == 'gradient_boosting':
            self.classifier = GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=42
            )

        else:
            raise ValueError(f"未知算法: {algorithm}")

        print("开始训练...")
        self.classifier.fit(self.X_train, self.y_train)
        print("训练完成!")

    def evaluate(self):
        """评估模型"""
        print("\n" + "="*60)
        print("模型评估")
        print("="*60)

        # 训练集准确率
        train_acc = self.classifier.score(self.X_train, self.y_train)
        print(f"\n训练集准确率: {train_acc:.4f} ({train_acc*100:.2f}%)")

        # 测试集准确率
        test_acc = self.classifier.score(self.X_test, self.y_test)
        print(f"测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")

        # 交叉验证
        cv_scores = cross_val_score(
            self.classifier, self.X_train, self.y_train, cv=5
        )
        print(f"\n5折交叉验证:")
        print(f"  平均准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # 详细分类报告
        y_pred = self.classifier.predict(self.X_test)
        target_names = [self.inverse_label_map[i] for i in range(len(self.label_map))]

        print("\n分类报告:")
        print(classification_report(self.y_test, y_pred, target_names=target_names))

        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)
        self.plot_confusion_matrix(cm, target_names)

    def plot_confusion_matrix(self, cm, labels):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        # 保存
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
        print(f"\n混淆矩阵已保存到: outputs/confusion_matrix.png")

    def save_model(self, output_path='models/gesture_classifier.pkl'):
        """保存模型"""
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)

        model_data = {
            'classifier': self.classifier,
            'label_map': self.label_map,
            'inverse_label_map': self.inverse_label_map
        }

        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n模型已保存到: {output_path}")

        # 保存模型信息
        info_path = output_path.with_suffix('.txt')
        with open(info_path, 'w') as f:
            f.write(f"手势分类器模型信息\n")
            f.write(f"="*60 + "\n\n")
            f.write(f"训练时间: {np.datetime64('now')}\n")
            f.write(f"算法: {type(self.classifier).__name__}\n")
            f.write(f"训练样本数: {len(self.X_train)}\n")
            f.write(f"测试样本数: {len(self.X_test)}\n")
            f.write(f"类别数: {len(self.label_map)}\n")
            f.write(f"类别: {list(self.label_map.keys())}\n\n")
            f.write(f"测试集准确率: {self.classifier.score(self.X_test, self.y_test):.4f}\n")

        print(f"模型信息已保存到: {info_path}")


def main():
    parser = argparse.ArgumentParser(description='手势分类器训练工具')
    parser.add_argument('--data', type=str, required=True,
                       help='训练数据JSON文件')
    parser.add_argument('--algorithm', type=str, default='random_forest',
                       choices=['random_forest', 'svm', 'knn', 'mlp', 'gradient_boosting'],
                       help='分类算法 (默认: random_forest)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='测试集比例 (默认: 0.2)')
    parser.add_argument('--output', type=str, default='models/gesture_classifier.pkl',
                       help='输出模型路径 (默认: models/gesture_classifier.pkl)')

    # 算法特定参数
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='树的数量 (Random Forest, Gradient Boosting)')
    parser.add_argument('--n-neighbors', type=int, default=5,
                       help='邻居数量 (KNN)')

    args = parser.parse_args()

    # 创建训练器
    trainer = GestureClassifierTrainer(args.data)

    # 加载数据
    trainer.load_data()

    # 划分数据集
    trainer.split_data(test_size=args.test_size)

    # 训练
    trainer.train(
        algorithm=args.algorithm,
        n_estimators=args.n_estimators,
        n_neighbors=args.n_neighbors
    )

    # 评估
    trainer.evaluate()

    # 保存模型
    trainer.save_model(args.output)

    print("\n训练完成!")


if __name__ == "__main__":
    main()
