
import dspy
# ======================
# 深度推理模組強化 (TAG模式)
# ======================
class QA_TAG(dspy.Signature):
    """技術問答專用TAG架構"""
    task_desc = """
    [Task] 技術支援問題解析
    [Action] 分析問題核心並比對知識庫
    [Goal] 提供精準且符合安全規範的解決方案
    """
    
    question = dspy.InputField(desc="用戶提出的原始技術問題")
    answer = dspy.OutputField(desc="基於知識庫的專業解答")

def connect_to_llm():
    """連接本地部署的Deepseek模型"""
    local_config = {
        "api_base": "http://localhost:11434/v1",
        "api_key": None,  # 明確設置None避免空值問題
        "model": "deepseek-r1:7b",
        "custom_llm_provider": "deepseek",
        "timeout": 30  # 增加超時設置
    }
    
    dspy.configure(lm=dspy.LM(**local_config))
    return dspy.Predict(QA_TAG)  # 返回TAG架構預測器

# ======================
# 提示詞增強模組
# ======================
def perform_augmentation(question: str) -> List[str]:
    """實施多維度提示詞增強"""
    augmentations = [
        # 技術領域強化
        f"[系統故障] {question} 請提供詳細日誌分析",
        f"[硬體問題] {question} 包含型號與錯誤代碼",
        
        # 多語言混合
        f"{question} (請用中文回答但保留關鍵技術英文術語)",
        
        # 情境擴展
        f"在企業級環境中，{question} 應如何處理？",
        f"若發生在雲端架構下，{question}"
    ]
    return augmentations