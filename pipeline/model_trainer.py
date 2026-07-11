from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

class IntelligenceModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Phân tách dữ liệu theo tỷ lệ 80/20, huấn luyện và đánh giá độ chính xác của phân loại.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        accuracy = self.model.score(X_test, y_test)
        return accuracy

    def predict_churn(self, target_features: np.ndarray) -> np.ndarray:
        """
        Dự đoán trạng thái hành vi của tệp khách hàng mục tiêu (1: Ở lại, 0: Rời bỏ).
        """
        return self.model.predict(target_features)
