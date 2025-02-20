import pandas as pd

def validate_augmented_data(filename):
    """數據質量驗證"""
    df = pd.read_csv(filename, header=None, names=['question', 'answer'])
    
    # 檢查空值
    print("空值統計：")
    print(df.isnull().sum())
    
    # 分析問題長度分布
    df['q_length'] = df['question'].apply(len)
    print("\n問題長度分布：")
    print(df['q_length'].describe())
