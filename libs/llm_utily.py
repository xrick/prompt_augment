import dspy

def connect_to_deepseek():
    local_config = {
        "api_base": "http://localhost:11434/v1",  # 注意需加/v1路徑
        "api_key": "NULL",  # 特殊標記用於跳過驗證
        "model": "deepseek-r1:7b",
        "custom_llm_provider": "deepseek"
    }

    dspy.configure(
        lm=dspy.LM(
            **local_config
        )
    )
    # 測試問答
    qa = dspy.Predict('question -> answer')
    return qa

def connect_to_llama():
    lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
    dspy.configure(lm=lm)

def connect_to_chatgpt():
    lm = dspy.LM('chatgpt-4', api_base='http://localhost:11434', api_key='')
    dspy.configure(lm=lm)

# Example usage
if __name__ == "__main__":
    qa_deepseek = connect_to_deepseek()
    connect_to_llama()
    connect_to_chatgpt()