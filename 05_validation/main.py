"""
Step 5: Validation Module
=========================

This module validates the generated schedule:
- Hard constraint checking
- Soft constraint evaluation
- Quality metrics calculation
- Validation report generation

Usage:
    python 05_validation/main.py
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class ValidationResult:
    constraint_name: str
    is_satisfied: bool
    violation_count: int
    violation_details: List[str]
    severity: str  # 'critical', 'warning', 'info'

class ScheduleValidator:
    def __init__(self, config_path: str):
        """Initialize schedule validator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create output directory
        os.makedirs(self.config['paths']['validation_dir'], exist_ok=True)
        
        # Load data
        self.classes_df = pd.read_csv(self.config['paths']['input_files']['classes'])
        self.teachers_df = pd.read_csv(self.config['paths']['input_files']['teachers'])
        self.rooms_df = pd.read_csv(self.config['paths']['input_files']['rooms'])
        self.curriculum_df = pd.read_csv(self.config['paths']['input_files']['curriculum'])
        self.availability_df = pd.read_csv(self.config['paths']['input_files']['teacher_availability'])
        
        # Load generated schedule
        self.schedule_df = pd.read_csv(self.config['paths']['output_files']['schedule'])
        
        # Initialize validation results
        self.validation_results = []
        self.quality_metrics = {}

    def validate_gender_compatibility(self) -> ValidationResult:
        """Validate gender compatibility between teachers and classes."""
        violations = []
        
        for _, entry in self.schedule_df.iterrows():
            # Get teacher and class info
            teacher_mask = self.teachers_df['teacher_id'] == entry['teacher_id']
            class_mask = self.classes_df['class_id'] == entry['class_id']
            
            if not teacher_mask.any():
                violations.append(f"Teacher {entry['teacher_id']} not found in teachers data")
                continue
            teacher = self.teachers_df[teacher_mask].iloc[0]
            
            if not class_mask.any():
                violations.append(f"Class {entry['class_id']} not found in classes data") 
                continue
            class_info = self.classes_df[class_mask].iloc[0]
            
            # Map class gender to teacher gender format for validation
            class_gender = class_info['gender']
            if class_gender == 'Boys':
                expected_teacher_gender = 'Male'
            elif class_gender == 'Girls':
                expected_teacher_gender = 'Female'
            else:
                expected_teacher_gender = class_gender
            
            if teacher['gender'] != expected_teacher_gender:
                violations.append(f"Teacher {entry['teacher_id']} ({teacher['gender']}) assigned to class {entry['class_id']} ({class_info['gender']})")
        
        return ValidationResult(
            constraint_name="Gender Compatibility",
            is_satisfied=len(violations) == 0,
            violation_count=len(violations),
            violation_details=violations,
            severity="critical"
        )

    def validate_teacher_availability(self) -> ValidationResult:
        """Validate teacher availability for assigned time slots."""
        violations = []
        
        for _, entry in self.schedule_df.iterrows():
            # Check teacher availability
            availability = self.availability_df[
                (self.availability_df['teacher_id'] == entry['teacher_id']) &
                (self.availability_df['day'] == entry['day']) &
                (self.availability_df['slot'] == entry['slot'])
            ]
            
            if len(availability) > 0 and not availability.iloc[0]['available']:
                violations.append(f"Teacher {entry['teacher_id']} not available on {entry['day']} slot {entry['slot']}")
        
        return ValidationResult(
            constraint_name="Teacher Availability",
            is_satisfied=len(violations) == 0,
            violation_count=len(violations),
            violation_details=violations,
            severity="critical"
        )

    def validate_room_capacity(self) -> ValidationResult:
        """Validate room capacity constraints."""
        violations = []
        
        # Check for double bookings
        room_slot_usage = {}
        for _, entry in self.schedule_df.iterrows():
            key = f"{entry['room_id']}_{entry['day']}_{entry['slot']}"
            if key in room_slot_usage:
                violations.append(f"Room {entry['room_id']} double booked on {entry['day']} slot {entry['slot']}")
            else:
                room_slot_usage[key] = entry
        
        return ValidationResult(
            constraint_name="Room Capacity",
            is_satisfied=len(violations) == 0,
            violation_count=len(violations),
            violation_details=violations,
            severity="critical"
        )

    def validate_room_type_requirements(self) -> ValidationResult:
        """Validate room type requirements for subjects."""
        violations = []
        
        for _, entry in self.schedule_df.iterrows():
            room_mask = self.rooms_df['room_id'] == entry['room_id']
            
            if not room_mask.any():
                violations.append(f"Room {entry['room_id']} not found in rooms data")
                continue
            room = self.rooms_df[room_mask].iloc[0]
            subject = entry['subject']
            
            # Check room type requirements
            room_requirements = self.config['room_requirements']
            
            if subject in room_requirements:
                requirements = room_requirements[subject]
                if room['room_type'] == 'Lab' and requirements['lab_ratio'] == 0:
                    violations.append(f"Subject {subject} assigned to Lab {entry['room_id']} but doesn't require lab")
                elif room['room_type'] == 'Playground' and requirements['playground_ratio'] == 0:
                    violations.append(f"Subject {subject} assigned to Playground {entry['room_id']} but doesn't require playground")
                elif room['room_type'] == 'Library' and requirements['library_ratio'] == 0:
                    violations.append(f"Subject {subject} assigned to Library {entry['room_id']} but doesn't require library")
        
        return ValidationResult(
            constraint_name="Room Type Requirements",
            is_satisfied=len(violations) == 0,
            violation_count=len(violations),
            violation_details=violations,
            severity="warning"
        )

    def validate_teacher_load_limits(self) -> ValidationResult:
        """Validate teacher load limits."""
        violations = []
        
        # Calculate teacher loads
        teacher_loads = self.schedule_df.groupby('teacher_id').size().to_dict()
        
        for teacher_id, load in teacher_loads.items():
            teacher_mask = self.teachers_df['teacher_id'] == teacher_id
            
            if not teacher_mask.any():
                violations.append(f"Teacher {teacher_id} not found in teachers data")
                continue
            teacher = self.teachers_df[teacher_mask].iloc[0]
            
            if load < teacher['min_periods']:
                violations.append(f"Teacher {teacher_id} has {load} periods (minimum: {teacher['min_periods']})")
            elif load > teacher['max_periods']:
                violations.append(f"Teacher {teacher_id} has {load} periods (maximum: {teacher['max_periods']})")
        
        return ValidationResult(
            constraint_name="Teacher Load Limits",
            is_satisfied=len(violations) == 0,
            violation_count=len(violations),
            violation_details=violations,
            severity="warning"
        )

    def validate_daily_class_requirement(self) -> ValidationResult:
        """Validate daily class requirement for teachers."""
        violations = []
        
        # Get teachers with daily class requirement
        daily_class_teachers = self.teachers_df[self.teachers_df['daily_class_required'] == True]['teacher_id'].tolist()
        
        for teacher_id in daily_class_teachers:
            teacher_schedule = self.schedule_df[self.schedule_df['teacher_id'] == teacher_id]
            
            # Check if teacher has classes every day
            days_with_classes = teacher_schedule['day'].unique()
            all_days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu'] 
            
            missing_days = [day for day in all_days if day not in days_with_classes]
            if missing_days:
                violations.append(f"Teacher {teacher_id} missing classes on: {', '.join(missing_days)}")
        
        return ValidationResult(
            constraint_name="Daily Class Requirement",
            is_satisfied=len(violations) == 0,
            violation_count=len(violations),
            violation_details=violations,
            severity="warning"
        )

    def validate_curriculum_coverage(self) -> ValidationResult:
        """Validate curriculum coverage for all classes."""
        violations = []
        
        for _, class_info in self.classes_df.iterrows():
            class_id = class_info['class_id']
            
            # Get required curriculum
            required_curriculum = self.curriculum_df[self.curriculum_df['class_id'] == class_id]
            
            # Get actual schedule
            actual_schedule = self.schedule_df[self.schedule_df['class_id'] == class_id]
            
            # Check each subject
            for _, subject_row in required_curriculum.iterrows():
                subject = subject_row['subject']
                required_periods = subject_row['periods_per_week']
                actual_periods = len(actual_schedule[actual_schedule['subject'] == subject])
                
                if actual_periods < required_periods:
                    violations.append(f"Class {class_id} missing {required_periods - actual_periods} periods for {subject}")
        
        return ValidationResult(
            constraint_name="Curriculum Coverage",
            is_satisfied=len(violations) == 0,
            violation_count=len(violations),
            violation_details=violations,
            severity="critical"
        )

    def calculate_quality_metrics(self):
        """Calculate quality metrics for the schedule."""
        print("Calculating quality metrics...")
        
        # Teacher utilization
        teacher_loads = self.schedule_df.groupby('teacher_id').size()
        self.quality_metrics['teacher_utilization'] = {
            'average_load': float(teacher_loads.mean()) if len(teacher_loads) > 0 else 0.0,
            'min_load': float(teacher_loads.min()) if len(teacher_loads) > 0 else 0.0,
            'max_load': float(teacher_loads.max()) if len(teacher_loads) > 0 else 0.0,
            'std_load': float(teacher_loads.std()) if len(teacher_loads) > 0 else 0.0,
            'teachers_used': int(len(teacher_loads)),
            'total_teachers': int(len(self.teachers_df))
        }
        
        # Room utilization
        room_usage = self.schedule_df.groupby('room_id').size()
        self.quality_metrics['room_utilization'] = {
            'average_usage': float(room_usage.mean()) if len(room_usage) > 0 else 0.0,
            'min_usage': float(room_usage.min()) if len(room_usage) > 0 else 0.0,
            'max_usage': float(room_usage.max()) if len(room_usage) > 0 else 0.0,
            'std_usage': float(room_usage.std()) if len(room_usage) > 0 else 0.0,
            'rooms_used': int(len(room_usage)),
            'total_rooms': int(len(self.rooms_df))
        }
        
        # Schedule balance (variance in daily periods per class)
        daily_periods = []
        for _, class_info in self.classes_df.iterrows():
            class_schedule = self.schedule_df[self.schedule_df['class_id'] == class_info['class_id']]
            daily_counts = class_schedule.groupby('day').size()
            daily_periods.extend(daily_counts.tolist())
        
        self.quality_metrics['schedule_balance'] = {
            'average_daily_periods': float(np.mean(daily_periods)) if daily_periods else 0.0,
            'std_daily_periods': float(np.std(daily_periods)) if daily_periods else 0.0,
            'min_daily_periods': float(np.min(daily_periods)) if daily_periods else 0.0,
            'max_daily_periods': float(np.max(daily_periods)) if daily_periods else 0.0
        }
        
        # Constraint satisfaction
        total_violations = sum(result.violation_count for result in self.validation_results)
        critical_violations = sum(result.violation_count for result in self.validation_results if result.severity == 'critical')
        
        self.quality_metrics['constraint_satisfaction'] = {
            'total_violations': int(total_violations),
            'critical_violations': int(critical_violations),
            'warning_violations': int(total_violations - critical_violations),
            'feasibility_score': float(max(0, 1 - (critical_violations / len(self.schedule_df)))) if len(self.schedule_df) > 0 else 0.0
        }
        
        # Overall quality score
        feasibility_score = self.quality_metrics['constraint_satisfaction']['feasibility_score']
        teacher_util_score = min(1.0, self.quality_metrics['teacher_utilization']['average_load'] / 20)  # Normalize to 20 periods
        room_util_score = min(1.0, self.quality_metrics['room_utilization']['average_usage'] / 35)  # Normalize to 35 periods
        balance_score = max(0, 1 - (self.quality_metrics['schedule_balance']['std_daily_periods'] / 5))  # Normalize to std=5
        
        self.quality_metrics['overall_quality'] = {
            'feasibility_score': float(feasibility_score),
            'teacher_utilization_score': float(teacher_util_score),
            'room_utilization_score': float(room_util_score),
            'balance_score': float(balance_score),
            'overall_score': float((feasibility_score * 0.4 + teacher_util_score * 0.2 + room_util_score * 0.2 + balance_score * 0.2))
        }

    def run_all_validations(self):
        """Run all validation checks."""
        print("Running validation checks...")
        
        # Hard constraints
        self.validation_results.append(self.validate_gender_compatibility())
        self.validation_results.append(self.validate_teacher_availability())
        self.validation_results.append(self.validate_room_capacity())
        self.validation_results.append(self.validate_curriculum_coverage())
        
        # Soft constraints
        self.validation_results.append(self.validate_room_type_requirements())
        self.validation_results.append(self.validate_teacher_load_limits())
        self.validation_results.append(self.validate_daily_class_requirement())
        
        # Calculate quality metrics
        self.calculate_quality_metrics()

    def save_validation_report(self):
        """Save validation report to JSON file."""
        report = {
            'validation_results': [
                {
                    'constraint_name': result.constraint_name,
                    'is_satisfied': result.is_satisfied,
                    'violation_count': result.violation_count,
                    'violation_details': result.violation_details,
                    'severity': result.severity
                }
                for result in self.validation_results
            ],
            'quality_metrics': self.quality_metrics,
            'summary': {
                'total_constraints': len(self.validation_results),
                'satisfied_constraints': sum(1 for r in self.validation_results if r.is_satisfied),
                'critical_violations': sum(r.violation_count for r in self.validation_results if r.severity == 'critical'),
                'warning_violations': sum(r.violation_count for r in self.validation_results if r.severity == 'warning'),
                'overall_feasibility': self.quality_metrics['overall_quality']['feasibility_score'],
                'overall_quality_score': self.quality_metrics['overall_quality']['overall_score']
            }
        }
        
        report_path = os.path.join(self.config['paths']['validation_dir'], 'validation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Validation report saved to: {report_path}")

    def print_validation_summary(self):
        """Print validation summary."""
        print("\nValidation Summary")
        print("=" * 50)
        
        for result in self.validation_results:
            status = " PASS" if result.is_satisfied else " FAIL"
            print(f"\n{result.constraint_name}: {status}")
            print(f"  Severity: {result.severity}")
            print(f"  Violations: {result.violation_count}")
            
            if result.violation_details and len(result.violation_details) <= 5:
                for violation in result.violation_details:
                    print(f"    - {violation}")
            elif result.violation_details:
                for violation in result.violation_details[:3]:
                    print(f"    - {violation}")
                print(f"    ... and {len(result.violation_details) - 3} more")
        
        print(f"\n Quality Metrics:")
        print(f"  Overall Quality Score: {self.quality_metrics['overall_quality']['overall_score']:.4f}")
        print(f"  Feasibility Score: {self.quality_metrics['overall_quality']['feasibility_score']:.4f}")
        print(f"  Teacher Utilization: {self.quality_metrics['teacher_utilization']['average_load']:.1f} periods")
        print(f"  Room Utilization: {self.quality_metrics['room_utilization']['average_usage']:.1f} periods")
        print(f"  Schedule Balance: {self.quality_metrics['schedule_balance']['average_daily_periods']:.1f} periods/day")

def main():
    """Main function for schedule validation."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'system_config.yaml')
    
    print("Starting Step 5: Schedule Validation")
    print("=" * 50)
    
    # Initialize validator
    validator = ScheduleValidator(config_path)
    
    # Run all validations
    validator.run_all_validations()
    
    # Save validation report
    validator.save_validation_report()
    
    # Print summary
    validator.print_validation_summary()
    
    print("\nStep 5 completed successfully!")
    print("   Ready for Step 6: Evaluation")

if __name__ == "__main__":
    main()
