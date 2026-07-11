import pandas as pd
import numpy as np
from pipeline.data_processor import CustomerDataProcessor
from pipeline.model_trainer import IntelligenceModelTrainer

def run_pipeline():
    print("====================================================")
    print("📊 INITIALIZING PREDICTIVE CUSTOMER INTEL PIPELINE  📊")
    print("====================================================\n")

    raw_mock_data = pd.DataFrame({
        'customer_id': [1001, 1002, 1003, 1004, 1005],
        'age': [24, np.nan, 35, 19, 42],
        'total_spend': [120.5, 450.0, np.nan, 25.0, 600.2],
        'days_active': [12, 45, 2, 1, 88]
    })

    labels = np.array([1, 1, 0, 0, 1])

    processor = CustomerDataProcessor()
    processed_df = processor.fit_transform(raw_mock_data)
    print("[Pipeline Log]: High-volume raw data transformation completed.")
    
    feature_cols = ['age', 'total_spend', 'days_active', 'engagement_score']
    features = processed_df[feature_cols].values
    
    trainer = IntelligenceModelTrainer()
    accuracy = trainer.train(features, labels)
    
    print("[Pipeline Log]: Predictive Classifier optimized via Random Forest algorithm.")
    print(f"[Accuracy Metric]: Evaluated General Model Accuracy: {accuracy * 100:.2f}%\n")
    
    sample_test = features[2].reshape(1, -1)
    prediction = trainer.predict_churn(sample_test)
    status = "Active (Ở lại)" if prediction[0] == 1 else "Churned (Nguy cơ rời bỏ)"
    print(f"🎯 [Prediction Output for Customer 1003]: System Risk Category -> {status}")
    print("====================================================")

if __name__ == "__main__":
    run_pipeline()
