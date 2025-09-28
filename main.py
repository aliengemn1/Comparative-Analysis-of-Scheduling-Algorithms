"""
School Schedule Generation System
==========================================

Main pipeline script that orchestrates the complete schedule generation process.

Usage:
    python main.py --config config/system_config.yaml
    python main.py --step 1  # Run only step 1
    python main.py --steps 1,2,3  # Run steps 1-3
"""

import os
import sys
import yaml
import argparse
import time
from typing import List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_step(step_number: int, config_path: str) -> bool:
    """Run a specific step of the pipeline."""
    step_modules = {
        1: "01_data_generation.main",
        2: "02_feature_engineering.main", 
        3: "03_model_training.main",
        4: "04_schedule_generation.main",
        5: "05_validation.main",
        6: "06_evaluation.main",
        7: "teacher_analysis.main"  # Add teacher analysis step
    }
    
    if step_number not in step_modules:
        print(f"Invalid step number: {step_number}")
        return False
    
    print(f"\nRunning Step {step_number}: {step_modules[step_number].split('.')[0].replace('_', ' ').title()}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Import and run the step module
        module_name = step_modules[step_number]
        module = __import__(module_name, fromlist=['main'])
        module.main()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nStep {step_number} completed successfully in {duration:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"\nStep {step_number} failed with error: {str(e)}")
        return False

def run_pipeline(config_path: str, steps: Optional[List[int]] = None) -> bool:
    """Run the complete automated schedule generation pipeline."""
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from: {config_path}")
    except Exception as e:
        print(f"Failed to load configuration: {str(e)}")
        return False
    
    # Determine which steps to run
    if steps is None:
        steps = list(range(1, 8))  # Run all steps including teacher analysis
    
    print(f"\nAutomated School Schedule Generation System")
    print(f" Running steps: {', '.join(map(str, steps))}")
    print("=" * 60)
    
    # Run each step
    successful_steps = []
    failed_steps = []
    
    for step in steps:
        success = run_step(step, config_path)
        if success:
            successful_steps.append(step)
        else:
            failed_steps.append(step)
            # Ask user if they want to continue
            print(f"\nStep {step} failed. Continuing with remaining steps...")
            # response = input(f"\nStep {step} failed. Continue with remaining steps? (y/n): ")
            # if response.lower() != 'y':
            #     break
    
    # Print final summary
    print(f"\nPipeline Execution Summary")
    print("=" * 60)
    print(f" Successful steps: {successful_steps}")
    if failed_steps:
        print(f" Failed steps: {failed_steps}")
    
    if len(successful_steps) == len(steps):
        print(f"\nAll steps completed successfully!")
        print(f"Generated schedule available at: {config['paths']['output_files']['schedule']}")
        print(f" Validation report available at: {config['paths']['validation_dir']}/validation_report.json")
        print(f" Evaluation report available at: {config['paths']['evaluation_dir']}/evaluation_report.json")
        return True
    else:
        print(f"\nPipeline completed with some failures")
        return False

def validate_config(config_path: str) -> bool:
    """Validate the configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['school', 'rooms', 'schedule', 'curriculum', 'teachers', 'ml', 'validation', 'evaluation', 'paths']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            print(f"Missing required configuration sections: {missing_sections}")
            return False
        
        # Check required paths
        required_paths = ['data_dir', 'features_dir', 'models_dir', 'schedules_dir', 'validation_dir', 'evaluation_dir']
        missing_paths = [path for path in required_paths if path not in config['paths']]
        
        if missing_paths:
            print(f"Missing required path configurations: {missing_paths}")
            return False
        
        print("Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {str(e)}")
        return False

def print_help():
    """Print help information."""
    print("""
 School Schedule Generation System
==========================================

This system generates school schedules using automated algorithms.

Usage:
    python main.py --config config/system_config.yaml
    python main.py --step 1
    python main.py --steps 1,2,3
    python main.py --validate-config
    python main.py --help

Options:
    --config PATH          Path to configuration file (required)
    --step N               Run only step N (1-7)
    --steps N,M,O         Run specific steps N, M, O
    --validate-config      Validate configuration file only
    --help                 Show this help message

Steps:
    1. Data Generation     - Generate base datasets (classes, teachers, rooms)
    2. Feature Engineering - Extract features for training
    3. Model Training     - Train models for schedule generation
    4. Schedule Generation - Generate schedules using trained models
    5. Validation         - Validate generated schedule quality
    6. Evaluation         - Compare and evaluate schedule performance
    7. Teacher Analysis   - Analyze teacher loads and generate substitution suggestions

Configuration:
    Edit config/system_config.yaml to customize:
    - School parameters (classes, teachers, rooms)
    - Schedule constraints and requirements
    - Model parameters
    - Output paths and file locations

Output:
    - Generated schedule: outputs/schedules/generated_schedule.csv
    - Validation report: outputs/validation/validation_report.json
    - Evaluation report: outputs/evaluation/evaluation_report.json
    - Quality metrics: outputs/evaluation/quality_metrics.csv
""")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Automated School Schedule Generation System')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--step', type=int, help='Run only specific step (1-6)')
    parser.add_argument('--steps', help='Run specific steps (comma-separated, e.g., 1,2,3)')
    parser.add_argument('--validate-config', action='store_true', help='Validate configuration file only')
    parser.add_argument('--help-full', action='store_true', help='Show detailed help')
    
    args = parser.parse_args()
    
    if args.help_full:
        print_help()
        return
    
    # Validate configuration
    if not validate_config(args.config):
        return
    
    if args.validate_config:
        print("Configuration validation completed")
        return
    
    # Determine steps to run
    steps = None
    if args.step:
        steps = [args.step]
    elif args.steps:
        try:
            steps = [int(s.strip()) for s in args.steps.split(',')]
        except ValueError:
            print("Invalid steps format. Use comma-separated numbers (e.g., 1,2,3)")
            return
    
    # Run pipeline
    success = run_pipeline(args.config, steps)
    
    if success:
        print(f"\nGeneration System completed successfully!")
    else:
        print(f"\nGeneration System completed with issues")
        sys.exit(1)

if __name__ == "__main__":
    main()
