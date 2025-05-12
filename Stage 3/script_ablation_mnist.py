# script/stage_3_script/script_ablation_mnist.py

import os
import sys

# Add project root to sys.path to allow imports from local_code
PROJECT_ROOT_FOR_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT_FOR_SCRIPT)

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting import Setting
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics

if __name__ == '__main__':
    DATA_PATH = os.path.join(PROJECT_ROOT_FOR_SCRIPT, 'data', 'stage_3_data') + os.sep
    ABLATION_RESULTS_PATH = os.path.join(PROJECT_ROOT_FOR_SCRIPT, 'result', 'stage_3_result', 'ablation_mnist') + os.sep
    os.makedirs(ABLATION_RESULTS_PATH, exist_ok=True)

    # --- Define Ablation Configurations ---
    # YOU MUST CAREFULLY ADJUST "model_config" FOR EACH ENTRY TO MATCH YOUR TABLE'S INTENDED ARCHITECTURE
    configurations = [
        {
            "id": "Baseline_2C_RRL_DR0.5_Ep30",
            "description": "Baseline: 2Conv(32,64 R/R), FC(128 R/Logits), DO(0.5)",
            "model_config": {
                "num_conv_layers": 2,
                "conv_out_channels": [32, 64],
                "conv_kernel_sizes": [5, 5],
                "conv_paddings": [2, 2],
                "conv_strides": [1, 1],
                "pool_kernel_sizes": [2, 2],
                "pool_strides": [2, 2],
                "conv_activation_strs": ["relu", "relu"],
                "fc_layer_sizes": [128],
                "fc_activation_strs": ["relu", None],
                "dropout_rate": 0.5,
                "verbose_run": True  # Enable per-epoch prints for the baseline
            },
            "max_epoch": 30
        },
        {
            "id": "Abl_T1_2C_Conv(S_S)_FC(S_Softmax)",
            "description": "Abl_T1: 2Conv(16,32 Sig/Sig), FC(100 Sig/Softmax), DO(0)",
            "model_config": {
                "num_conv_layers": 2,
                "conv_out_channels": [16, 32],  # Example, adjust to your table's intent
                "conv_activation_strs": ["sigmoid", "sigmoid"],
                "fc_layer_sizes": [100],
                "fc_activation_strs": ["sigmoid", "softmax"],
                "dropout_rate": 0.0,
                "verbose_run": False
            },
            "max_epoch": 15
        },
        {
            "id": "Abl_T2_2C_Conv(S_T)_FC(T_Softmax)",
            "description": "Abl_T2: 2Conv(16,32 Sig/Tanh), FC(100 Tanh/Softmax), DO(0)",
            "model_config": {
                "num_conv_layers": 2,
                "conv_out_channels": [16, 32],
                "conv_activation_strs": ["sigmoid", "tanh"],
                "fc_layer_sizes": [100],
                "fc_activation_strs": ["tanh", "softmax"],
                "dropout_rate": 0.0,
                "verbose_run": False
            },
            "max_epoch": 15
        },
        {
            "id": "Abl_T3_2C_Conv(R_T)_FC(R_Softmax)",
            "description": "Abl_T3: 2Conv(16,32 ReLU/Tanh), FC(100 ReLU/Softmax), DO(0)",
            "model_config": {
                "num_conv_layers": 2,
                "conv_out_channels": [16, 32],
                "conv_activation_strs": ["relu", "tanh"],
                "fc_layer_sizes": [100],
                "fc_activation_strs": ["relu", "softmax"],
                "dropout_rate": 0.0,
                "verbose_run": False
            },
            "max_epoch": 15
        },
        {
            "id": "Abl_T4_2C_Conv(R_T)_FC(T_Softmax)",
            "description": "Abl_T4: 2Conv(16,32 ReLU/Tanh), FC(100 Tanh/Softmax), DO(0)",
            "model_config": {
                "num_conv_layers": 2,
                "conv_out_channels": [16, 32],
                "conv_activation_strs": ["relu", "tanh"],
                "fc_layer_sizes": [100],
                "fc_activation_strs": ["tanh", "softmax"],
                "dropout_rate": 0.0,
                "verbose_run": False
            },
            "max_epoch": 15
        },
        {
            "id": "Abl_T5_3C_Conv(R_T_T)_FC(P_Softmax)",
            "description": "Abl_T5: 3Conv(16,32,64 R/T/T), FC(100 PReLU/Softmax), DO(0)",
            "model_config": {
                "num_conv_layers": 3,
                "conv_out_channels": [16, 32, 64],
                "conv_activation_strs": ["relu", "tanh", "tanh"],
                "fc_layer_sizes": [100],
                "fc_activation_strs": ["prelu", "softmax"],
                "dropout_rate": 0.0,
                "verbose_run": False
            },
            "max_epoch": 15
        }
    ]

    data_obj = Dataset_Loader('MNIST_Dataset_for_Ablation', 'Loads MNIST data for ablation studies.')
    data_obj.dataset_source_file_name = 'MNIST'
    data_obj.dataset_source_folder_path = DATA_PATH

    # IMPORTANT: Ensure data_obj.load() is called if your Setting.py doesn't do it.
    # If Setting.py expects data_obj to be a ready-to-use PyTorch Dataset,
    # then Dataset_Loader must inherit from torch.utils.data.Dataset and load data in __init__ or be called.
    # My Dataset_Loader above is now a simple file loader utility.
    # Its load() method is expected to be called by Setting.py: loaded_dict = self.dataset.load()
    # And then Setting.py should pass this loaded_dict to method_obj.data.

    evaluate_obj = Evaluate_Metrics('Metrics_Evaluator_for_Ablation', 'Evaluates model performance.')

    print("************ Starting MNIST Ablation Study ************")
    all_run_performances = []

    for i, current_config_setup in enumerate(configurations):
        config_id = current_config_setup["id"]
        config_desc = current_config_setup["description"]
        model_specific_config = current_config_setup["model_config"]
        run_max_epoch = current_config_setup.get("max_epoch", Method_CNN_MNIST.max_epoch)

        print(f"\n--- Running Configuration {i + 1}/{len(configurations)}: {config_id} ---")
        print(f"Description: {config_desc}")

        method_obj = Method_CNN_MNIST(
            mName=config_id,
            mDescription=config_desc,
            config=model_specific_config,
        )
        method_obj.max_epoch = run_max_epoch
        # Pass verbosity from main config to method object for its internal prints
        method_obj.verbose_run_prints = model_specific_config.get('verbose_run', False)

        result_obj = Result_Saver('Ablation_Result_Saver', 'Saves prediction results for ablation run.')
        result_obj.result_destination_folder_path = ABLATION_RESULTS_PATH
        result_obj.result_destination_file_name = f'predictions_{config_id}'

        setting_obj = Setting(f'Ablation_Setting_{config_id}', 'Configures and runs one ablation experiment.')
        # This prepare method needs to ensure that method_obj gets its data.
        # Typically, setting_obj.dataset.load() is called inside setting_obj.load_run_save_evaluate(),
        # and the result is passed to method_obj.data.
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)

        try:
            evaluator_name_to_print = getattr(setting_obj.evaluate, 'mName', setting_obj.evaluate.__class__.__name__)
            method_name_to_print = getattr(setting_obj.method, 'mName', setting_obj.method.__class__.__name__)
            print(
                f"Executing: Evaluator='{evaluator_name_to_print}', Method='{method_name_to_print}', ConfigID='{config_id}', Epochs={method_obj.max_epoch}")

            performance_metrics = setting_obj.load_run_save_evaluate()

            print("--- Performance ---")
            current_performance_summary = {"ID": config_id, "Description": config_desc}
            if isinstance(performance_metrics, dict):
                for key, val in performance_metrics.items():
                    metric_key = key.replace(" score", "").replace(" ", "")
                    print(f"{key:15s}: {val}")
                    current_performance_summary[metric_key] = val
            else:
                print(performance_metrics)
                current_performance_summary["RawResult"] = performance_metrics
            all_run_performances.append(current_performance_summary)

        except Exception as e:
            print(f"!!!!!! ERROR during configuration: {config_id} !!!!!!")
            print(f"Error details: {e}")
            import traceback

            traceback.print_exc()
            all_run_performances.append({"ID": config_id, "Description": config_desc, "Error": str(e)})

        print(f"--- Finished Configuration: {config_id} ---")

    print("\n\n************ Ablation Study Summary ************")
    header_desc_len = 70
    row_format_string = "| {:<{desc_len}} | {:<10} | {:<10} | {:<10} | {:<10} |"
    header = row_format_string.format(
        "Description", "Accuracy", "Precision", "Recall", "F1 Score", desc_len=header_desc_len
    )
    print(header)
    print("-" * len(header))

    for res in all_run_performances:
        desc_to_print = res.get('Description', 'N/A')
        if len(desc_to_print) > header_desc_len:
            desc_to_print = desc_to_print[:header_desc_len - 3] + "..."

        if "Error" in res:
            print(row_format_string.format(desc_to_print, 'ERROR', 'ERROR', 'ERROR', 'ERROR', desc_len=header_desc_len))
        else:
            acc = res.get('Accuracy', res.get('accuracy', 'N/A'))
            prec = res.get('Precision', res.get('precision', 'N/A'))
            rec = res.get('Recall', res.get('recall', 'N/A'))
            f1 = res.get('F1', res.get('F1Score', res.get('F1 Score', res.get('F1 score', res.get('f1', 'N/A')))))

            acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
            prec_str = f"{prec:.4f}" if isinstance(prec, float) else str(prec)
            rec_str = f"{rec:.4f}" if isinstance(rec, float) else str(rec)
            f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)

            print(row_format_string.format(desc_to_print, acc_str, prec_str, rec_str, f1_str, desc_len=header_desc_len))

    print("-" * len(header))
    print("************ Ablation Study Finished ************")