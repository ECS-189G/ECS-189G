# script/stage_3_script/script_ablation_orl.py

import os
import sys

PROJECT_ROOT_FOR_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT_FOR_SCRIPT)

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL  # The configurable version
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting import Setting
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics

if __name__ == '__main__':
    DATA_PATH = os.path.join(PROJECT_ROOT_FOR_SCRIPT, 'data', 'stage_3_data') + os.sep
    ABLATION_RESULTS_PATH = os.path.join(PROJECT_ROOT_FOR_SCRIPT, 'result', 'stage_3_result', 'ablation_orl') + os.sep
    os.makedirs(ABLATION_RESULTS_PATH, exist_ok=True)

    # --- Define ORL Ablation Configurations ---
    # Base parameters for ORL from previous successful/default model
    orl_base_model_params = {
        "conv1_out_channels": 32, "conv1_kernel_size": 5, "conv1_stride": 1, "conv1_padding": 2,
        "conv2_out_channels": 64, "conv2_kernel_size": 5, "conv2_stride": 1, "conv2_padding": 2,
        "pool_kernel_size": 2, "pool_stride": 2,
        "activation_str": "relu",
        "fc_hidden_sizes": [256],  # One hidden FC layer of 256 units
        "use_batchnorm": True,
        "dropout_rate": 0.5,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,  # Default with Adam
        "optimizer_type": "adam",
        "lr_scheduler_step_size": 15,  # For StepLR
        "lr_scheduler_gamma": 0.2,  # For StepLR
        "verbose_run": False  # Suppress per-epoch prints for multiple runs by default
    }

    configurations = [
        {
            "id": "ORL_Baseline_Ep50",
            "description": "Baseline (2Conv+2Pool+1HiddenFC256; ReLU; lr=1e-3; BN; DO=0.5)",
            "model_config": {**orl_base_model_params},  # Use a copy of base
            "max_epoch": 50
        },
        {
            "id": "ORL_Abl_ConvFilters_64_16",
            "description": "Conv1->64 filters, Conv2->16 filters",
            "model_config": {**orl_base_model_params,
                             "conv1_out_channels": 64,
                             "conv2_out_channels": 16},
            "max_epoch": 50
        },
        {
            "id": "ORL_Abl_MaxEpochs_40",
            "description": "Max epochs -> 40",
            "model_config": {**orl_base_model_params},
            "max_epoch": 40
        },
        {
            "id": "ORL_Abl_LR_5e-3",
            "description": "Learning rate -> 5e-3",
            "model_config": {**orl_base_model_params, "learning_rate": 5e-3},
            "max_epoch": 50
        },
        {
            "id": "ORL_Abl_ConvKernels_C1(3x3_s1p1)_C2(7x7_s2p?)",  # Padding for Conv2 with 7x7s2 needs care
            "description": "Conv1 k3s1p1; Conv2 k7s2pX (padding needs adjustment)",
            "model_config": {**orl_base_model_params,
                             "conv1_kernel_size": 3, "conv1_padding": 1, "conv1_stride": 1,
                             "conv2_kernel_size": 7, "conv2_padding": 3, "conv2_stride": 2  # Example padding for s2,k7
                             },  # Note: Changing kernel/stride/pad WILL change flattened_features size.
            # The Method_CNN_ORL above calculates this dynamically.
            "max_epoch": 50
        },
        {
            "id": "ORL_Abl_NoBatchNorm",
            "description": "Remove BatchNorm",
            "model_config": {**orl_base_model_params, "use_batchnorm": False},
            "max_epoch": 50
        },
        {
            "id": "ORL_Abl_RemoveFC3",  # This means no hidden FC layers
            "description": "Remove FC3 (flatten -> FC4 output directly)",
            "model_config": {**orl_base_model_params,
                             "fc_hidden_sizes": [],  # Empty list means no hidden FC layers
                             "remove_fc3": True  # A specific flag Method_CNN_ORL can use
                             },
            "max_epoch": 50
        }
    ]

    # --- Setup shared objects ---
    data_obj = Dataset_Loader('ORL_Dataset_for_Ablation', 'Loads ORL data.')
    data_obj.dataset_source_file_name = 'ORL'  # Adjust if your file is ORL.pkl
    data_obj.dataset_source_folder_path = DATA_PATH

    evaluate_obj = Evaluate_Metrics('Metrics_Evaluator_for_Ablation_ORL', '')

    print("************ Starting ORL Ablation Study ************")
    all_run_performances = []

    for i, current_config_setup in enumerate(configurations):
        config_id = current_config_setup["id"]
        config_desc = current_config_setup["description"]
        model_specific_config = current_config_setup["model_config"]
        run_max_epoch = current_config_setup.get("max_epoch", Method_CNN_ORL.max_epoch)

        print(f"\n--- Running ORL Configuration {i + 1}/{len(configurations)}: {config_id} ---")
        print(f"Description: {config_desc}")

        method_obj = Method_CNN_ORL(
            mName=config_id,
            mDescription=config_desc,
            config=model_specific_config
        )
        method_obj.max_epoch = run_max_epoch
        method_obj.verbose_run_prints = model_specific_config.get('verbose_run', False)  # Control prints

        result_obj = Result_Saver('Ablation_ORL_Result_Saver', '')
        result_obj.result_destination_folder_path = ABLATION_RESULTS_PATH
        result_obj.result_destination_file_name = f'orl_predictions_{config_id}'

        setting_obj = Setting(f'Ablation_ORL_Setting_{config_id}', '')
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
            print(f"!!!!!! ERROR during ORL configuration: {config_id} !!!!!!")
            print(f"Error details: {e}")
            import traceback

            traceback.print_exc()
            all_run_performances.append({"ID": config_id, "Description": config_desc, "Error": str(e)})

        print(f"--- Finished ORL Configuration: {config_id} ---")

    # --- Print Summary Table ---
    print("\n\n************ ORL Ablation Study Summary ************")
    # (Summary table printing logic similar to script_ablation_mnist.py)
    header_desc_len = 75  # Adjusted for longer ORL descriptions
    row_format_string = "| {:<{desc_len}} | {:<10} | {:<10} | {:<10} | {:<10} |"
    header = row_format_string.format("Description", "Accuracy", "Precision", "Recall", "F1 Score",
                                      desc_len=header_desc_len)
    print(header);
    print("-" * len(header))
    for res in all_run_performances:
        desc_to_print = res.get('Description', 'N/A')[:header_desc_len - 3] + "..." if len(
            res.get('Description', 'N/A')) > header_desc_len else res.get('Description', 'N/A')
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
    print("************ ORL Ablation Study Finished ************")