# script/stage_3_script/script_ablation_cifar.py

import os
import sys

PROJECT_ROOT_FOR_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT_FOR_SCRIPT)

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting import Setting
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics

if __name__ == '__main__':
    DATA_PATH = os.path.join(PROJECT_ROOT_FOR_SCRIPT, 'data', 'stage_3_data') + os.sep
    ABLATION_RESULTS_PATH = os.path.join(PROJECT_ROOT_FOR_SCRIPT, 'result', 'stage_3_result', 'ablation_cifar') + os.sep
    os.makedirs(ABLATION_RESULTS_PATH, exist_ok=True)

    # Base config for CIFAR-10 (matches Method_CNN_CIFAR default_config and your table's baseline)
    cifar_base_model_config = {
        "num_conv_blocks": 3,
        "block_channels": [[32, 64], [128, 128], [256, 256]],
        "fc_layer_sizes": [1024, 512],
        "activation_str": "relu",
        "use_batchnorm_overall": True,  # Baseline has BN
        "fc_dropout_rate": 0.0,  # Baseline from table description has no FC dropout
        "learning_rate": 1e-3,  # Baseline LR from table
        "optimizer_type": "adam", "weight_decay": 1e-4,
        "lr_scheduler_type": "cosine", "T_max_cosine": 30, "eta_min_cosine": 1e-6,  # Assuming 30 epochs for baseline
        "verbose_run": False
    }

    configurations = [
        {
            "id": "CIFAR_Baseline_Ep30",
            "description": "Baseline (6Conv+3Pool+BNs+3FC; ReLU; lr=1e-3)",
            # Your table says "2BN", Method_CNN_CIFAR adds BN per conv layer if enabled
            "model_config": {**cifar_base_model_config, "verbose_run": True},
            "max_epoch": 30
        },
        {
            "id": "CIFAR_Abl_NoBatchNorm",
            "description": "Remove all BatchNorm",
            "model_config": {**cifar_base_model_config, "use_batchnorm_overall": False},
            "max_epoch": 30
        },
        {
            "id": "CIFAR_Abl_Dropout0.5_After_FC1",
            "description": "Add Dropout(0.5) after FC1",
            "model_config": {**cifar_base_model_config,
                             "fc_dropout_rate": 0.5,
                             "fc_dropout_after_fc1_only": True  # Flag used in Method_CNN_CIFAR
                             },
            "max_epoch": 30
        },
        {
            "id": "CIFAR_Abl_Optimizer_SGD",
            "description": "Optimizer -> SGD (lr=0.01, momentum=0.9)",
            "model_config": {**cifar_base_model_config,
                             "optimizer_type": "sgd",
                             "learning_rate": 0.01,
                             "momentum": 0.9,
                             "weight_decay": 1e-4,  # Common with SGD
                             "lr_scheduler_type": "step",  # StepLR often paired with SGD
                             "lr_scheduler_step_size": 10,
                             "lr_scheduler_gamma": 0.1
                             },
            "max_epoch": 30
        },
        {
            "id": "CIFAR_Abl_IncDepth_4thConvBlock512",
            "description": "Increase depth: add 4th conv block (512, 512 filters)",
            "model_config": {**cifar_base_model_config,
                             "num_conv_blocks": 4,
                             # Add channel specs for the 4th block
                             "block_channels": [[32, 64], [128, 128], [256, 256], [512, 512]],
                             },
            "max_epoch": 30  # May need more epochs for deeper model
        },
        {
            "id": "CIFAR_Abl_RemoveConv6",  # Assuming Conv6 is the second conv in the 3rd block
            "description": "Remove final convolution layer (Conv6 of baseline)",
            "model_config": {**cifar_base_model_config, "remove_final_conv_layer": True},
            "max_epoch": 30
        }
    ]

    data_obj = Dataset_Loader('CIFAR_Dataset_for_Ablation', 'Loads CIFAR-10 data.')
    data_obj.dataset_source_file_name = 'CIFAR'  # MODIFIED based on user's filename
    data_obj.dataset_source_folder_path = DATA_PATH

    evaluate_obj = Evaluate_Metrics('Metrics_Evaluator_for_Ablation_CIFAR', '')

    print("************ Starting CIFAR-10 Ablation Study ************")
    all_run_performances = []

    # --- Loop, method instantiation, run, and summary printing logic ---
    # This part is identical in structure to script_ablation_mnist.py and script_ablation_orl.py
    for i, current_config_setup in enumerate(configurations):
        config_id = current_config_setup["id"]
        config_desc = current_config_setup["description"]
        model_specific_config = current_config_setup["model_config"]
        run_max_epoch = current_config_setup.get("max_epoch", Method_CNN_CIFAR.max_epoch)

        print(f"\n--- Running CIFAR Configuration {i + 1}/{len(configurations)}: {config_id} ---")
        print(f"Description: {config_desc}")

        method_obj = Method_CNN_CIFAR(
            mName=config_id,
            mDescription=config_desc,
            config=model_specific_config
        )
        method_obj.max_epoch = run_max_epoch
        method_obj.verbose_run_prints = model_specific_config.get('verbose_run', False)

        result_obj = Result_Saver('Ablation_CIFAR_Result_Saver', '')
        result_obj.result_destination_folder_path = ABLATION_RESULTS_PATH
        result_obj.result_destination_file_name = f'cifar_predictions_{config_id}'

        setting_obj = Setting(f'Ablation_CIFAR_Setting_{config_id}', '')
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)

        try:
            eval_name = getattr(setting_obj.evaluate, 'mName', setting_obj.evaluate.__class__.__name__)
            meth_name = getattr(setting_obj.method, 'mName', setting_obj.method.__class__.__name__)
            print(
                f"Executing: Evaluator='{eval_name}', Method='{meth_name}', ConfigID='{config_id}', Epochs={method_obj.max_epoch}")

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
            print(f"!!!!!! ERROR during CIFAR configuration: {config_id} !!!!!!")
            print(f"Error details: {e}")
            import traceback

            traceback.print_exc()
            all_run_performances.append({"ID": config_id, "Description": config_desc, "Error": str(e)})

        print(f"--- Finished CIFAR Configuration: {config_id} ---")

    # --- Print Summary Table ---
    print("\n\n************ CIFAR-10 Ablation Study Summary ************")
    header_desc_len = 80
    row_format_string = "| {:<{desc_len}} | {:<10} | {:<10} | {:<10} | {:<10} |"
    header = row_format_string.format("Description", "Accuracy", "Precision", "Recall", "F1 Score",
                                      desc_len=header_desc_len)
    print(header);
    print("-" * len(header))
    for res in all_run_performances:
        desc_to_print = res.get('Description', 'N/A')
        if len(desc_to_print) > header_desc_len: desc_to_print = desc_to_print[:header_desc_len - 3] + "..."

        if "Error" in res:
            print(row_format_string.format(desc_to_print, 'ERROR', 'ERROR', 'ERROR', 'ERROR', desc_len=header_desc_len))
        else:
            acc = res.get('Accuracy', 'N/A');
            prec = res.get('Precision', 'N/A');
            rec = res.get('Recall', 'N/A')
            f1 = res.get('F1', res.get('F1Score', res.get('F1 score', 'N/A')))
            acc_s, prec_s, rec_s, f1_s = (f"{x:.4f}" if isinstance(x, float) else str(x) for x in [acc, prec, rec, f1])
            print(row_format_string.format(desc_to_print, acc_s, prec_s, rec_s, f1_s, desc_len=header_desc_len))
    print("-" * len(header));
    print("************ CIFAR-10 Ablation Study Finished ************")