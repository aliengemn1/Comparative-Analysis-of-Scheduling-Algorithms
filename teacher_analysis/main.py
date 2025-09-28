"""
Teacher Load Analysis and Substitution Module
============================================

This module analyzes teacher loads and generates substitution suggestions
for different scheduling algorithms (greedy, iterative, automated).

Usage:
    python teacher_analysis/main.py
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Tuple
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TeacherLoadAnalyzer:
    def __init__(self, config_path: str):
        """Initialize teacher load analyzer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Create output directory
        os.makedirs('outputs/teacher_analysis', exist_ok=True)
        
        # Load base data
        self.classes_df = pd.read_csv(self.config['paths']['input_files']['classes'])
        self.teachers_df = pd.read_csv(self.config['paths']['input_files']['teachers'])
        self.rooms_df = pd.read_csv(self.config['paths']['input_files']['rooms'])
        self.curriculum_df = pd.read_csv(self.config['paths']['input_files']['curriculum'])
        self.availability_df = pd.read_csv(self.config['paths']['input_files']['teacher_availability'])

    def analyze_teacher_loads(self, schedule_df: pd.DataFrame, algorithm_name: str) -> Dict[str, Any]:
        """Analyze teacher loads for a given schedule."""
        print(f"Analyzing teacher loads for {algorithm_name} algorithm...")
        
        # Calculate teacher loads
        teacher_loads = schedule_df.groupby('teacher_id').size().to_dict()
        
        # Get teacher details
        teacher_details = {}
        for _, teacher in self.teachers_df.iterrows():
            teacher_id = teacher['teacher_id']
            load = teacher_loads.get(teacher_id, 0)
            
            teacher_details[teacher_id] = {
                'name': teacher['name'],
                'gender': teacher['gender'],
                'primary_subject': teacher['primary_subject'],
                'secondary_subjects': teacher['secondary_subjects'].split(',') if pd.notna(teacher['secondary_subjects']) else [],
                'home_building': teacher['home_building'],
                'daily_class_required': teacher['daily_class_required'],
                'min_periods': teacher['min_periods'],
                'max_periods': teacher['max_periods'],
                'current_load': load,
                'load_status': self._get_load_status(load, teacher['min_periods'], teacher['max_periods'])
            }
        
        # Calculate statistics
        loads = list(teacher_loads.values())
        stats = {
            'total_teachers': len(self.teachers_df),
            'teachers_used': len(teacher_loads),
            'teachers_unused': len(self.teachers_df) - len(teacher_loads),
            'average_load': np.mean(loads) if loads else 0,
            'min_load': min(loads) if loads else 0,
            'max_load': max(loads) if loads else 0,
            'std_load': np.std(loads) if loads else 0,
            'underloaded_teachers': sum(1 for t in teacher_details.values() if t['load_status'] == 'underloaded'),
            'overloaded_teachers': sum(1 for t in teacher_details.values() if t['load_status'] == 'overloaded'),
            'optimal_teachers': sum(1 for t in teacher_details.values() if t['load_status'] == 'optimal')
        }
        
        return {
            'algorithm': algorithm_name,
            'statistics': stats,
            'teacher_details': teacher_details,
            'teacher_loads': teacher_loads
        }

    def _get_load_status(self, current_load: int, min_periods: int, max_periods: int) -> str:
        """Determine teacher load status."""
        if current_load < min_periods:
            return 'underloaded'
        elif current_load > max_periods:
            return 'overloaded'
        else:
            return 'optimal'

    def generate_substitution_suggestions(self, schedule_df: pd.DataFrame, 
                                        teacher_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate substitution suggestions for absent teachers."""
        print(f"Generating substitution suggestions for {teacher_analysis['algorithm']}...")
        
        algorithm_name = teacher_analysis['algorithm']
        teacher_details = teacher_analysis['teacher_details']
        
        # Simulate absent teachers (10% of teachers)
        total_teachers = len(teacher_details)
        num_absent = max(1, int(total_teachers * 0.1))
        absent_teachers = random.sample(list(teacher_details.keys()), num_absent)
        
        substitution_suggestions = {}
        
        for absent_teacher_id in absent_teachers:
            absent_teacher = teacher_details[absent_teacher_id]
            
            # Find affected classes
            affected_classes = schedule_df[
                schedule_df['teacher_id'] == absent_teacher_id
            ]['class_id'].unique().tolist()
            
            suggestions = []
            
            for class_id in affected_classes:
                class_info = self.classes_df[self.classes_df['class_id'] == class_id].iloc[0]
                
                # Get class subjects
                class_subjects = schedule_df[
                    schedule_df['class_id'] == class_id
                ]['subject'].unique().tolist()
                
                for subject in class_subjects:
                    # Find potential substitutes
                    potential_substitutes = self._find_potential_substitutes(
                        absent_teacher_id, subject, class_info, teacher_details, schedule_df
                    )
                    
                    if potential_substitutes:
                        suggestions.append({
                            'class_id': class_id,
                            'subject': subject,
                            'substitutes': potential_substitutes[:3]  # Top 3 suggestions
                        })
            
            substitution_suggestions[absent_teacher_id] = {
                'absent_teacher': {
                    'name': absent_teacher['name'],
                    'primary_subject': absent_teacher['primary_subject'],
                    'current_load': absent_teacher['current_load']
                },
                'affected_classes': affected_classes,
                'suggestions': suggestions
            }
        
        return {
            'algorithm': algorithm_name,
            'absent_teachers': absent_teachers,
            'substitution_suggestions': substitution_suggestions
        }

    def _find_potential_substitutes(self, absent_teacher_id: str, subject: str, 
                                   class_info: pd.Series, teacher_details: Dict[str, Any],
                                   schedule_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find potential substitute teachers for a specific assignment."""
        substitutes = []
        
        for teacher_id, teacher in teacher_details.items():
            if teacher_id == absent_teacher_id:
                continue
            
            # Check gender compatibility
            if teacher['gender'] != class_info['gender']:
                continue
            
            # Check subject capability
            can_teach_subject = (
                teacher['primary_subject'] == subject or 
                subject in teacher['secondary_subjects']
            )
            
            if not can_teach_subject:
                continue
            
            # Check if teacher is not overloaded
            if teacher['load_status'] == 'overloaded':
                continue
            
            # Check daily class requirement
            if teacher['daily_class_required'] and teacher['current_load'] == 0:
                continue
            
            # Calculate compatibility score
            compatibility_score = self._calculate_compatibility_score(
                teacher, subject, class_info, schedule_df
            )
            
            substitutes.append({
                'teacher_id': teacher_id,
                'name': teacher['name'],
                'primary_subject': teacher['primary_subject'],
                'current_load': teacher['current_load'],
                'compatibility_score': compatibility_score,
                'reason': self._get_substitution_reason(teacher, subject, class_info)
            })
        
        # Sort by compatibility score
        substitutes.sort(key=lambda x: x['compatibility_score'], reverse=True)
        return substitutes

    def _calculate_compatibility_score(self, teacher: Dict[str, Any], subject: str,
                                     class_info: pd.Series, schedule_df: pd.DataFrame) -> float:
        """Calculate compatibility score for substitute teacher."""
        score = 0.0
        
        # Primary subject match
        if teacher['primary_subject'] == subject:
            score += 0.4
        
        # Building match
        if teacher['home_building'] == class_info['building']:
            score += 0.2
        
        # Load balance (prefer teachers with moderate load)
        current_load = teacher['current_load']
        if 15 <= current_load <= 20:
            score += 0.2
        elif 10 <= current_load <= 25:
            score += 0.1
        
        # Availability (check if teacher has free slots)
        teacher_schedule = schedule_df[schedule_df['teacher_id'] == teacher['teacher_id']]
        if len(teacher_schedule) < teacher['max_periods']:
            score += 0.2
        
        return score

    def _get_substitution_reason(self, teacher: Dict[str, Any], subject: str, 
                               class_info: pd.Series) -> str:
        """Get reason for substitution suggestion."""
        reasons = []
        
        if teacher['primary_subject'] == subject:
            reasons.append("Primary subject match")
        
        if teacher['home_building'] == class_info['building']:
            reasons.append("Same building")
        
        if teacher['current_load'] < teacher['min_periods']:
            reasons.append("Underloaded")
        elif teacher['current_load'] < teacher['max_periods']:
            reasons.append("Available capacity")
        
        return "; ".join(reasons) if reasons else "General availability"

    def generate_algorithm_comparison(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparison between different algorithms."""
        print("Generating algorithm comparison...")
        
        comparison = {
            'algorithms': [],
            'summary_statistics': {},
            'recommendations': []
        }
        
        for analysis in analyses:
            algorithm_name = analysis['algorithm']
            stats = analysis['statistics']
            
            comparison['algorithms'].append({
                'name': algorithm_name,
                'statistics': stats
            })
        
        # Calculate summary statistics
        algorithms = comparison['algorithms']
        comparison['summary_statistics'] = {
            'best_utilization': max(algorithms, key=lambda x: x['statistics']['average_load']),
            'most_balanced': min(algorithms, key=lambda x: x['statistics']['std_load']),
            'fewest_underloaded': min(algorithms, key=lambda x: x['statistics']['underloaded_teachers']),
            'fewest_overloaded': min(algorithms, key=lambda x: x['statistics']['overloaded_teachers'])
        }
        
        # Generate recommendations
        comparison['recommendations'] = self._generate_recommendations(algorithms)
        
        return comparison

    def _generate_recommendations(self, algorithms: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on algorithm comparison."""
        recommendations = []
        
        # Find best algorithm for each metric
        best_utilization = max(algorithms, key=lambda x: x['statistics']['average_load'])
        most_balanced = min(algorithms, key=lambda x: x['statistics']['std_load'])
        
        recommendations.append(f"Best teacher utilization: {best_utilization['name']} ({best_utilization['statistics']['average_load']:.1f} periods/teacher)")
        recommendations.append(f"Most balanced load distribution: {most_balanced['name']} (std: {most_balanced['statistics']['std_load']:.1f})")
        
        # Check for underloaded teachers
        for algo in algorithms:
            underloaded = algo['statistics']['underloaded_teachers']
            if underloaded > 0:
                recommendations.append(f"{algo['name']}: {underloaded} teachers are underloaded - consider reassigning periods")
        
        # Check for overloaded teachers
        for algo in algorithms:
            overloaded = algo['statistics']['overloaded_teachers']
            if overloaded > 0:
                recommendations.append(f"{algo['name']}: {overloaded} teachers are overloaded - consider reducing their load")
        
        return recommendations

    def save_analysis_results(self, analyses: List[Dict[str, Any]], 
                            substitutions: List[Dict[str, Any]], 
                            comparison: Dict[str, Any]):
        """Save all analysis results to files."""
        print("Saving teacher analysis results...")
        
        # Save individual algorithm analyses
        for analysis in analyses:
            algorithm_name = analysis['algorithm'].lower().replace(' ', '_')
            
            # Save teacher loads
            teacher_loads_df = pd.DataFrame([
                {
                    'teacher_id': teacher_id,
                    'name': details['name'],
                    'gender': details['gender'],
                    'primary_subject': details['primary_subject'],
                    'current_load': details['current_load'],
                    'min_periods': details['min_periods'],
                    'max_periods': details['max_periods'],
                    'load_status': details['load_status']
                }
                for teacher_id, details in analysis['teacher_details'].items()
            ])
            
            teacher_loads_df.to_csv(
                f'outputs/teacher_analysis/{algorithm_name}_teacher_loads.csv',
                index=False
            )
            
            # Save statistics
            with open(f'outputs/teacher_analysis/{algorithm_name}_statistics.json', 'w') as f:
                json.dump(analysis['statistics'], f, indent=2)
        
        # Save substitution suggestions
        for substitution in substitutions:
            algorithm_name = substitution['algorithm'].lower().replace(' ', '_')
            
            substitution_data = []
            for teacher_id, suggestions in substitution['substitution_suggestions'].items():
                for suggestion in suggestions['suggestions']:
                    for substitute in suggestion['substitutes']:
                        substitution_data.append({
                            'absent_teacher_id': teacher_id,
                            'absent_teacher_name': suggestions['absent_teacher']['name'],
                            'class_id': suggestion['class_id'],
                            'subject': suggestion['subject'],
                            'substitute_teacher_id': substitute['teacher_id'],
                            'substitute_teacher_name': substitute['name'],
                            'compatibility_score': substitute['compatibility_score'],
                            'reason': substitute['reason']
                        })
            
            if substitution_data:
                substitution_df = pd.DataFrame(substitution_data)
                substitution_df.to_csv(
                    f'outputs/teacher_analysis/{algorithm_name}_substitutions.csv',
                    index=False
                )
        
        # Save comparison results
        with open('outputs/teacher_analysis/algorithm_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"Analysis results saved to: outputs/teacher_analysis/")

    def print_analysis_summary(self, analyses: List[Dict[str, Any]], 
                             substitutions: List[Dict[str, Any]], 
                             comparison: Dict[str, Any]):
        """Print summary of teacher analysis results."""
        print("\nTeacher Load Analysis Summary")
        print("=" * 60)
        
        for analysis in analyses:
            stats = analysis['statistics']
            print(f"\n{analysis['algorithm']} Algorithm:")
            print(f"  Teachers Used: {stats['teachers_used']}/{stats['total_teachers']}")
            print(f"  Average Load: {stats['average_load']:.1f} periods")
            print(f"  Load Range: {stats['min_load']}-{stats['max_load']} periods")
            print(f"  Load Balance: {stats['std_load']:.1f} std dev")
            print(f"  Underloaded: {stats['underloaded_teachers']} teachers")
            print(f"  Overloaded: {stats['overloaded_teachers']} teachers")
            print(f"  Optimal: {stats['optimal_teachers']} teachers")
        
        print(f"\nSubstitution Analysis:")
        for substitution in substitutions:
            algorithm_name = substitution['algorithm']
            num_absent = len(substitution['absent_teachers'])
            total_suggestions = sum(len(s['suggestions']) for s in substitution['substitution_suggestions'].values())
            print(f"  {algorithm_name}: {num_absent} absent teachers, {total_suggestions} substitution suggestions")
        
        print(f"\nAlgorithm Comparison:")
        summary = comparison['summary_statistics']
        print(f"  Best Utilization: {summary['best_utilization']['name']}")
        print(f"  Most Balanced: {summary['most_balanced']['name']}")
        print(f"  Fewest Underloaded: {summary['fewest_underloaded']['name']}")
        print(f"  Fewest Overloaded: {summary['fewest_overloaded']['name']}")
        
        print(f"\nRecommendations:")
        for rec in comparison['recommendations']:
            print(f"  - {rec}")

def main():
    """Main function for teacher load analysis."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'system_config.yaml')
    
    print("Starting Teacher Load Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = TeacherLoadAnalyzer(config_path)
    
    # Load schedules from different algorithms
    schedules = {}
    
    # Try to load generated schedule
    try:
        ml_schedule = pd.read_csv('outputs/schedules/ml_generated_schedule.csv')
        schedules['Generated'] = ml_schedule
        print("Loaded generated schedule")
    except FileNotFoundError:
        print("Generated schedule not found, skipping...")
    
    # Try to load greedy schedule
    try:
        greedy_schedule = pd.read_csv('data/results/greedy/schedule_greedy.csv')
        schedules['Greedy'] = greedy_schedule
        print("Loaded greedy schedule")
    except FileNotFoundError:
        print("Greedy schedule not found, skipping...")
    
    # Try to load iterative schedule
    try:
        iterative_schedule = pd.read_csv('data/results/iterative/schedule_iterative.csv')
        schedules['Iterative'] = iterative_schedule
        print("Loaded iterative schedule")
    except FileNotFoundError:
        print("Iterative schedule not found, skipping...")
    
    if not schedules:
        print("No schedules found to analyze!")
        return
    
    # Analyze each schedule
    analyses = []
    substitutions = []
    
    for algorithm_name, schedule_df in schedules.items():
        # Analyze teacher loads
        analysis = analyzer.analyze_teacher_loads(schedule_df, algorithm_name)
        analyses.append(analysis)
        
        # Generate substitution suggestions
        substitution = analyzer.generate_substitution_suggestions(schedule_df, analysis)
        substitutions.append(substitution)
    
    # Generate comparison
    comparison = analyzer.generate_algorithm_comparison(analyses)
    
    # Save results
    analyzer.save_analysis_results(analyses, substitutions, comparison)
    
    # Print summary
    analyzer.print_analysis_summary(analyses, substitutions, comparison)
    
    print("\nTeacher Load Analysis completed successfully!")

if __name__ == "__main__":
    main()
