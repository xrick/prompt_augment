
import dspy
from typing import List
from abc import ABC, abstractmethod
import csv
# ======================
class AugmentationStrategy(ABC):
    @abstractmethod
    def augment(self, question: str) -> str:
        pass

class TAGStrategy(AugmentationStrategy):
    def augment(self, question: str) -> str:
        task = "技術支援問題解析"
        goal = "提供精準且符合安全規範的解決方案"
        guidance = "分析問題核心並比對知識庫"
        return f"[Task] {task}\n[Goal] {goal}\n[Guidance] {guidance}\n[Question] {question}"

class CIDIStrategy(AugmentationStrategy):
    def augment(self, question: str) -> str:
        context = "企業級環境"
        instructions = "請提供詳細日誌分析"
        details = "包含型號與錯誤代碼"
        inputs = question
        return f"[Context] {context}\n[Instructions] {instructions}\n[Details] {details}\n[Inputs] {inputs}"

def perform_augmentation(question: str, strategy_type: str) -> str:
    """實施多維度提示詞增強"""
    strategies = {
        "TAG": TAGStrategy(),
        "CIDI": CIDIStrategy()
    }
    strategy = strategies.get(strategy_type)
    if not strategy:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    return strategy.augment(question)
    


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

def save_aug_data_to_csv(aug_questions: List[str], answer: str, filename="augmented_dataset.csv"):
    """保存增強數據到CSV"""
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for q in aug_questions:
            writer.writerow([q, answer])


def augment_prompt(q_lst: list, v_lst: list):
    """全自動提示詞增強流水線"""
    for index in range(len(q_lst)):
        try:
            question = q_lst[index]
            answer = v_lst[index]
            
            # 執行多層次增強
            aug_prompts = perform_augmentation(question)
            
            # 過濾無效增強
            valid_aug = [p for p in aug_prompts if len(p) > 15]  # 長度過濾
            
            # 持久化儲存
            save_aug_data_to_csv(valid_aug, answer)
            
            print(f"已處理第{index+1}筆，生成{len(valid_aug)}個增強提示")
            
        except Exception as e:
            print(f"處理索引{index}時發生錯誤: {str(e)}")
            continue