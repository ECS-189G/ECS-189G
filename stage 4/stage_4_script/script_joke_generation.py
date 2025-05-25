import os
import numpy as np
import torch
import traceback # 用于打印详细的错误信息

from sympy import false

# 假设这些本地模块已正确实现并在 PYTHONPATH 中
# 你需要确保这些文件的实际路径能被 Python 解释器找到
from local_code.stage_4_code.Joke_Dataset_Loader import JokeDataLoader
from local_code.stage_4_code.Joke_Different_Method import JokeRNNMethod
from local_code.stage_4_code.Joke_Result_Saver import Result_Saver
from local_code.stage_4_code.Joke_Setting import Setting
from local_code.stage_4_code.Joke_Evaluate_Metrics import Evaluate_Metrics

# ---- 配置参数部分 ------------------------------------------------------
SEED = 2
TRAINING_MODE = False # True: 训练模型; False: 加载预训练模型进行生成
START_PROMPT = ("Write it down right away" )  # 用于文本生成的初始提示

# 路径配置
# 注意：'../../' 这样的相对路径是相对于执行脚本时的“当前工作目录”
# 为了更稳健，可以考虑使用基于脚本文件位置的绝对路径，例如：
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # 获取当前脚本文件的目录
# ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')) # 假设项目根目录是两级以上
# DATA_FOLDER_PATH = os.path.join(ROOT_DIR, 'data', 'stage_4_data', 'text_generation')
# RESULT_FOLDER_PATH = os.path.join(ROOT_DIR, 'result', 'stage_4_result')
DATA_FOLDER_PATH = r'../../data/stage_4_data/text_generation/' # 你原来的相对路径
RESULT_FOLDER_PATH = r'../../result/stage_4_result/'       # 你原来的相对路径


DATASET_SOURCE_FILE_NAME = 'data'  # 例如 'jokes_dataset.txt' 或如果 'data' 是一个特殊标识符
RESULT_DESTINATION_FILE_NAME = 'generated_jokes.txt'

# JokeRNNMethod 内部模型保存路径的默认值 (如果你的 JokeRNNMethod 类有这个默认设置)
# 如果 JokeRNNMethod 类没有此默认值，它可能需要在实例化时或通过属性设置
# 或者，如果 Setting 类负责模型路径，则此处的注释可能不适用
# MODEL_SAVE_PATH_FOR_RNN = "./joke-weights.pth" # 这是 JokeRNNMethod 内部的默认值
# ---------------------------------------------------------------------------

def set_seeds(seed_value):
    """设置随机种子以保证可复现性"""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value) # 为所有GPU设置种子
        # 以下两行可能会稍微降低性能，但能增强CUDA运算的确定性
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to: {seed_value}")

def main():
    """主函数，用于编排笑话生成流程"""
    set_seeds(SEED)

    print(f"{'*'*12} Start {'*'*12}")

    # ---- 对象初始化部分 (使用位置参数传递名称和描述) -----------------------
    print("Initializing objects...")
    try:
        data_obj = JokeDataLoader(
            'Joke Dataset Loader',
            'Loads and prepares joke data for text generation.'
        )
        data_obj.dataset_source_folder_path = DATA_FOLDER_PATH
        data_obj.dataset_source_file_name = DATASET_SOURCE_FILE_NAME

        # **load your data now** so vocab_size, encoded_data, etc. get populated
        if hasattr(data_obj, 'load'):
            data_obj.load()
        else:
            raise RuntimeError(
                "JokeDataLoader has no load() method — make sure you call whatever initializer populates vocab_size, encoded_data, vocab_to_int, and int_to_vocab.")

        method_obj = JokeRNNMethod(
            'Joke RNN Model',
            'Recurrent Neural Network for joke generation.',
            data_obj
        )

        # 如果 JokeRNNMethod 需要显式设置模型保存/加载路径，可以在这里设置：
        # method_obj.model_save_path = MODEL_SAVE_PATH_FOR_RNN # 或者从配置中读取

        result_obj = Result_Saver(
            'Result Saver',                                         # 位置参数1: name
            'Saves generated text to a file.'                       # 位置参数2: description
        )
        result_obj.result_destination_folder_path = RESULT_FOLDER_PATH
        result_obj.result_destination_file_name = RESULT_DESTINATION_FILE_NAME

        setting_obj = Setting(
            'Joke Generation Orchestrator',                         # 位置参数1: name
            'Orchestrates the joke generation pipeline.'            # 位置参数2: description
        )

        evaluate_obj = Evaluate_Metrics(
            'Evaluation Metrics Calculator',                        # 位置参数1: name
            'Handles evaluation of generated jokes (if applicable).' # 位置参数2: description
        )
        print("Objects initialized successfully.")
    except TypeError as te:
        print(f"TypeError during object initialization: {te}")
        print("Please check the __init__ method signatures of your custom classes.")
        print("They might not match the positional arguments being passed for name and description.")
        traceback.print_exc()
        return # 退出 main 函数
    except Exception as e:
        print(f"An unexpected error occurred during object initialization: {e}")
        traceback.print_exc()
        return # 退出 main 函数
    # ---------------------------------------------------------------------------

    # ---- 运行部分 -------------------------------------------------------------
    try:
        print("\nPreparing setup...")
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)

        print("\nSetup Summary:")
        setting_obj.print_setup_summary()

        print(f"\nRunning model (Training Mode: {TRAINING_MODE})...")
        generated_text = setting_obj.load_run_save_evaluate(
            start_tokens=START_PROMPT,
            training=TRAINING_MODE
        )

        print(f"\n{'*'*12} Generated Text {'*'*12}")
        if isinstance(generated_text, str) and generated_text.startswith("Error:"):
            print(f"Generation process reported an error: {generated_text}")
        elif generated_text and isinstance(generated_text, str): # 确保是字符串且非空
            print(generated_text)
        else:
            print("No text was generated, or an unexpected result type was returned.")

    except FileNotFoundError as fnf_error:
        print(f"FileNotFoundError during run section: {fnf_error}")
        print("This often means a model file or data file was not found. If not training, ensure a pre-trained model exists at the expected path.")
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred during the running section: {e}")
        traceback.print_exc()
    finally:
        print(f"\n{'*'*12} Finish {'*'*12}")
    # ---------------------------------------------------------------------------

if __name__ == '__main__':
    main()


