


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import argparse
from pathlib import Path
import yaml
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd, description=""):
    
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Success: {description}")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error: {e}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False

def run_preprocessing_experiment(args):
    
    logger.info("="*50)
    logger.info("PREPROCESSING EXPERIMENT")
    logger.info("="*50)
    
    cmd = [
        'python', 'scripts/preprocess_data.py',
        '--input_path', args.input_data,
        '--output_dir', args.output_dir + '/processed_data',
        '--format', args.data_format,
        '--signal_length', str(args.signal_length),
        '--n_workers', str(args.n_workers)
    ]
    
    if args.quality_check:
        cmd.append('--quality_check')
    
    return run_command(cmd, "Data preprocessing")

def run_dual_stream_experiment(args):
    
    logger.info("="*50)
    logger.info("DUAL STREAM ENCODER EXPERIMENT")
    logger.info("="*50)
    
    config_path = Path(args.config_dir) / 'dual_stream_config.yaml'
    model_dir = Path(args.output_dir) / 'dual_stream_model'
    
    
    cmd = [
        'python', 'scripts/train.py',
        '--config', str(config_path),
        '--output_dir', str(model_dir)
    ]
    
    success = run_command(cmd, "Dual stream encoder training")
    
    if success and args.evaluate:
        
        cmd = [
            'python', 'scripts/evaluate.py',
            '--config', str(config_path),
            '--model_path', str(model_dir / 'final_model.pth'),
            '--data_dir', args.output_dir + '/processed_data',
            '--output_dir', str(model_dir / 'evaluation'),
            '--visualize'
        ]
        
        run_command(cmd, "Dual stream encoder evaluation")
    
    return success

def run_multi_expert_experiment(args):
    
    logger.info("="*50)
    logger.info("MULTI EXPERT MODEL EXPERIMENT")
    logger.info("="*50)
    
    config_path = Path(args.config_dir) / 'multi_expert_config.yaml'
    model_dir = Path(args.output_dir) / 'multi_expert_model'
    
    
    cmd = [
        'python', 'scripts/train.py',
        '--config', str(config_path),
        '--output_dir', str(model_dir)
    ]
    
    success = run_command(cmd, "Multi expert model training")
    
    if success and args.evaluate:
        
        cmd = [
            'python', 'scripts/evaluate.py',
            '--config', str(config_path),
            '--model_path', str(model_dir / 'final_model.pth'),
            '--data_dir', args.output_dir + '/processed_data',
            '--output_dir', str(model_dir / 'evaluation'),
            '--visualize'
        ]
        
        run_command(cmd, "Multi expert model evaluation")
    
    return success

def run_transformer_experiment(args):
    
    logger.info("="*50)
    logger.info("TRANSFORMER MODEL EXPERIMENT")
    logger.info("="*50)
    
    config_path = Path(args.config_dir) / 'transformer_config.yaml'
    model_dir = Path(args.output_dir) / 'transformer_model'
    
    
    cmd = [
        'python', 'scripts/train.py',
        '--config', str(config_path),
        '--output_dir', str(model_dir)
    ]
    
    success = run_command(cmd, "Transformer model training")
    
    if success and args.evaluate:
        
        cmd = [
            'python', 'scripts/evaluate.py',
            '--config', str(config_path),
            '--model_path', str(model_dir / 'final_model.pth'),
            '--data_dir', args.output_dir + '/processed_data',
            '--output_dir', str(model_dir / 'evaluation'),
            '--visualize'
        ]
        
        run_command(cmd, "Transformer model evaluation")
    
    return success

def run_comparison_experiment(args):
    
    logger.info("="*50)
    logger.info("MODEL COMPARISON EXPERIMENT")
    logger.info("="*50)
    
    
    models = ['dual_stream', 'multi_expert', 'transformer']
    results = {}
    
    for model_name in models:
        eval_dir = Path(args.output_dir) / f'{model_name}_model' / 'evaluation'
        result_file = eval_dir / 'single_task_results.json' if model_name != 'multi_expert' else eval_dir / 'multi_task_results.json'
        
        if result_file.exists():
            import json
            with open(result_file, 'r') as f:
                results[model_name] = json.load(f)
    
    
    comparison_report = generate_comparison_report(results)
    
    
    report_path = Path(args.output_dir) / 'comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write(comparison_report)
    
    logger.info(f"Comparison report saved to {report_path}")

def generate_comparison_report(results):
    
    report = []
    report.append("="*60)
    report.append("MODEL COMPARISON REPORT")
    report.append("="*60)
    report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    
    single_task_models = [k for k, v in results.items() if k != 'multi_expert' and 'metrics' in v]
    
    if single_task_models:
        report.append("SINGLE TASK MODELS:")
        report.append("-" * 40)
        report.append(f"{'Model':<15} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Pearson':<8}")
        report.append("-" * 40)
        
        for model in single_task_models:
            metrics = results[model]['metrics']
            report.append(f"{model:<15} {metrics['mae']:<8.4f} {metrics['rmse']:<8.4f} {metrics['r2']:<8.4f} {metrics['pearson']:<8.4f}")
    
    
    if 'multi_expert' in results:
        report.append("")
        report.append("MULTI EXPERT MODEL:")
        report.append("-" * 50)
        report.append(f"{'Task':<8} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Pearson':<8}")
        report.append("-" * 50)
        
        for task, data in results['multi_expert'].items():
            if 'metrics' in data:
                metrics = data['metrics']
                report.append(f"{task:<8} {metrics['mae']:<8.4f} {metrics['rmse']:<8.4f} {metrics['r2']:<8.4f} {metrics['pearson']:<8.4f}")
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Run PPG experiments')
    parser.add_argument('--input_data', default='./sample_data.csv', help='Input data path')
    parser.add_argument('--data_format', choices=['csv', 'npy', 'hdf5'], default='csv', help='Input data format')
    parser.add_argument('--output_dir', default='./experiment_outputs', help='Output directory')
    parser.add_argument('--config_dir', default='./configs', help='Config directory')
    parser.add_argument('--signal_length', type=int, default=600, help='Signal length')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--quality_check', action='store_true', help='Enable quality check')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation after training')
    parser.add_argument('--experiments', nargs='+', 
                       choices=['preprocess', 'dual_stream', 'multi_expert', 'transformer', 'compare', 'all'],
                       default=['all'], help='Experiments to run')
    
    args = parser.parse_args()
    
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    
    if 'all' in args.experiments:
        experiments = ['preprocess', 'dual_stream', 'multi_expert', 'transformer', 'compare']
    else:
        experiments = args.experiments
    
    logger.info(f"Running experiments: {experiments}")
    
    success_count = 0
    total_count = 0
    
    
    if 'preprocess' in experiments:
        total_count += 1
        if run_preprocessing_experiment(args):
            success_count += 1
    
    if 'dual_stream' in experiments:
        total_count += 1
        if run_dual_stream_experiment(args):
            success_count += 1
    
    if 'multi_expert' in experiments:
        total_count += 1
        if run_multi_expert_experiment(args):
            success_count += 1
    
    if 'transformer' in experiments:
        total_count += 1
        if run_transformer_experiment(args):
            success_count += 1
    
    if 'compare' in experiments:
        run_comparison_experiment(args)
    
    
    logger.info("="*50)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*50)
    logger.info(f"Successful experiments: {success_count}/{total_count}")
    logger.info(f"Results saved to: {args.output_dir}")
    
    if success_count == total_count:
        logger.info("All experiments completed successfully!")
    else:
        logger.warning(f"{total_count - success_count} experiments failed. Check logs for details.")

if __name__ == '__main__':
    main()
