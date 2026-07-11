import pandas as pd
import numpy as np

class CustomerDataProcessor:
    def __init__(self):
        print("[Processor Init]: Core Data Manipulation Layer Activated.")

    def fit_transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Tự động làm sạch, xử lý dữ liệu khuyết thiếu (Imputation) 
        và thực hiện Feature Engineering để tối ưu hóa thuộc tính mô hình.
        """
        df = raw_data.copy()
        if 'age' in df.columns:
            df['age'] = df['age'].fillna(df['age'].mean())
        if 'total_spend' in df.columns:
            df['total_spend'] = df['total_spend'].fillna(0)
        df['engagement_score'] = (df['total_spend'] * 0.7) + (df['days_active'] * 0.3)
        
        return df
