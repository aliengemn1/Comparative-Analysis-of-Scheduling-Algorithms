"""
Step 4: Optimized Schedule Generation Module
===========================================

This module generates schedules using trained models with performance optimizations:
- Cached lookups and precomputed data structures
- Vectorized operations where possible
- Reduced redundant calculations
- Optimized teacher and room selection algorithms

Usage:
    python 04_schedule_generation/main.py
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Any, Tuple, Optional, Set
import random
from dataclasses import dataclass
import json
from collections import defaultdict, Counter

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class ScheduleEntry:
    class_id: str
    teacher_id: str
    subject: str
    room_id: str
    day: str
    slot: int
    confidence: float

class OptimizedScheduleGenerator:
    def __init__(self, config_path: str):
        """Initialize schedule generator with configuration and precompute data structures."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Create output directory
        os.makedirs(self.config['paths']['schedules_dir'], exist_ok=True)
        
        # Load data
        self._load_data()
        
        # Load trained models
        self._load_models()
        
        # Precompute data structures for faster lookups
        self._precompute_data_structures()
        
        # Initialize schedule tracking
        self.schedule = []
        self.teacher_loads = defaultdict(int)
        self.room_usage = defaultdict(set)  # room_id -> set of (day, slot) tuples
        self.teacher_schedule = defaultdict(set)  # teacher_id -> set of (day, slot) tuples
        self.class_schedules = {}

    def _load_data(self):
        """Load all CSV data files."""
        self.classes_df = pd.read_csv(self.config['paths']['input_files']['classes'])
        self.teachers_df = pd.read_csv(self.config['paths']['input_files']['teachers'])
        self.rooms_df = pd.read_csv(self.config['paths']['input_files']['rooms'])
        self.curriculum_df = pd.read_csv(self.config['paths']['input_files']['curriculum'])
        self.availability_df = pd.read_csv(self.config['paths']['input_files']['teacher_availability'])

    def _load_models(self):
        """Load all trained models."""
        self.models = {}
        models_dir = self.config['paths']['models_dir']
        self.models['teacher_assignment'] = joblib.load(os.path.join(models_dir, 'teacher_assignment_model.pkl'))
        self.models['schedule_quality'] = joblib.load(os.path.join(models_dir, 'schedule_quality_model.pkl'))
        self.models['constraint_satisfaction'] = joblib.load(os.path.join(models_dir, 'constraint_satisfaction_model.pkl'))
        
        # Load features for reference
        self.compatibility_features = pd.read_csv(
            os.path.join(self.config['paths']['features_dir'], 'teacher_class_compatibility.csv')
        )
        
        # Load subject combinations matrix
        self.subject_combinations = self.config.get('subject_combinations', {})

    def _precompute_data_structures(self):
        """Precompute data structures for faster lookups during scheduling."""
        print("Precomputing data structures for optimization...")
        
        # Create teacher lookup dictionaries
        self.teacher_by_id = {row['teacher_id']: row for _, row in self.teachers_df.iterrows()}
        self.room_by_id = {row['room_id']: row for _, row in self.rooms_df.iterrows()}
        self.class_by_id = {row['class_id']: row for _, row in self.classes_df.iterrows()}
        
        # Precompute teacher availability as nested dict for O(1) lookup
        self.teacher_availability = defaultdict(lambda: defaultdict(lambda: defaultdict(bool)))
        for _, row in self.availability_df.iterrows():
            self.teacher_availability[row['teacher_id']][row['day']][row['slot']] = row['available']
        
        # Precompute teachers by subject and gender compatibility
        self.teachers_by_subject_gender = defaultdict(list)
        for _, teacher in self.teachers_df.iterrows():
            subjects = [teacher['primary_subject']]
            if pd.notna(teacher['secondary_subjects']):
                subjects.extend(teacher['secondary_subjects'].split(','))
            
            for subject in subjects:
                subject = subject.strip()
                for gender in ['Boys', 'Girls', 'Mixed']:
                    teacher_gender = 'Male' if gender == 'Boys' else 'Female' if gender == 'Girls' else teacher['gender']
                    if teacher['gender'] == teacher_gender or gender == 'Mixed':
                        self.teachers_by_subject_gender[(subject, gender)].append(teacher)
        
        # Precompute rooms by building and type
        self.rooms_by_building_type = defaultdict(list)
        for _, room in self.rooms_df.iterrows():
            self.rooms_by_building_type[(room['building'], room['room_type'])].append(room)
            # Also add to general building category
            self.rooms_by_building_type[(room['building'], 'any')].append(room)
        
        # Precompute curriculum requirements by class
        self.curriculum_by_class = defaultdict(dict)
        for _, row in self.curriculum_df.iterrows():
            self.curriculum_by_class[row['class_id']][row['subject']] = row['periods_per_week']
        
        # Precompute subject sequences
        self.subject_sequences = self._precompute_subject_sequences()
        
        # Cache room suitability for subjects
        self.room_subject_cache = {}
        for _, room in self.rooms_df.iterrows():
            for subject in set(self.curriculum_df['subject']):
                self.room_subject_cache[(room['room_id'], subject)] = self._is_room_suitable_for_subject(room, subject)

    def _precompute_subject_sequences(self) -> Dict:
        """Precompute subject sequences for all grades."""
        sequences = {}
        
        # High school (G7-9)
        sequences['high'] = {
            'arabic': {'suggested_slots': [1, 6], 'priority': 1},
            'english': {'suggested_slots': [2, 7], 'priority': 1},
            'math': {'suggested_slots': [1, 2,3,6,8], 'priority': 1},
            'science': {'suggested_slots': [1, 2,3,6,8], 'priority': 1},
            'library': {'suggested_slots': [1, 5, 2, 4], 'priority': 3},
            'islamic': {'suggested_slots': [5, 6], 'priority': 1},
            'social': {'suggested_slots': [5, 6], 'priority': 3},
            'history': {'suggested_slots': [5, 6], 'priority': 2},
            'geography': {'suggested_slots': [5, 6], 'priority': 2},
            'pe': {'suggested_slots': [3, 4], 'priority': 4},
            'art': {'suggested_slots': [6, 7,8], 'priority': 4},
            'music': {'suggested_slots': [6,7, 8], 'priority': 4},
            'computer': {'suggested_slots': [4, 5, 6], 'priority': 3},
            'french': {'suggested_slots': [6, 7, 8], 'priority': 3},
            'chinese': {'suggested_slots': [6, 5, 4], 'priority': 3},
        }
        
        # Middle school (G4-6)
        sequences['middle'] = {
            'arabic': {'suggested_slots': [1, 6, 7], 'priority': 1},
            'english': {'suggested_slots': [2, 7], 'priority': 1},
            'math': {'suggested_slots': [1, 2,3,6,8], 'priority': 1},
            'science': {'suggested_slots': [1, 2,3,6,8], 'priority': 2},
            'library': {'suggested_slots': [2, 5, 1], 'priority': 2},
            'islamic': {'suggested_slots': [5], 'priority': 2},
            'social': {'suggested_slots': [5, 6], 'priority': 3},
            'history': {'suggested_slots': [5, 6], 'priority': 3},
            'geography': {'suggested_slots': [5, 6], 'priority': 3},
            'pe': {'suggested_slots': [3, 4], 'priority': 4},
            'art': {'suggested_slots': [6, 7], 'priority': 4},
            'music': {'suggested_slots': [7, 8], 'priority': 4},
            'computer': {'suggested_slots': [4, 5], 'priority': 3},
            'french': {'suggested_slots': [6, 7, 8], 'priority': 3},
        }
        
        # Elementary (G1-3)
        sequences['elementary'] = {
            'arabic': {'suggested_slots': [1, 2, 3, 6, 7], 'priority': 1},
            'english': {'suggested_slots': [2, 3, 7], 'priority': 1},
            'math': {'suggested_slots': [1, 2,3,6,8], 'priority': 1},
            'science': {'suggested_slots': [1, 2,3,6,8], 'priority': 2},
            'library': {'suggested_slots': [1, 2, 5], 'priority': 2},
            'pe': {'suggested_slots': [3, 4], 'priority': 4},
            'art': {'suggested_slots': [6, 7], 'priority': 4},
            'music': {'suggested_slots': [7, 8], 'priority': 4},
            'computer': {'suggested_slots': [4, 5], 'priority': 3},
            'french': {'suggested_slots': [6, 7, 8], 'priority': 3},
        }
        
        return sequences

    def get_available_teachers_fast(self, class_info: pd.Series, subject: str) -> List[pd.Series]:
        """Fast teacher lookup using precomputed data structures."""
        class_gender = class_info['gender']
        
        # Get from precomputed cache
        available_teachers = self.teachers_by_subject_gender.get((subject, class_gender), [])
        
        # Filter by capacity and sort by current load
        teachers_with_capacity = []
        for teacher in available_teachers:
            current_load = self.teacher_loads[teacher['teacher_id']]
            if current_load < teacher['max_periods']:
                # Add load info for sorting
                teacher_with_load = teacher.copy()
                teacher_with_load['current_load'] = current_load
                teachers_with_capacity.append(teacher_with_load)
        
        # Sort by current load (prioritize teachers with lower load)
        teachers_with_capacity.sort(key=lambda x: x['current_load'])
        
        return teachers_with_capacity

    def get_available_rooms_fast(self, class_info: pd.Series, subject: str) -> List[pd.Series]:
        """Fast room lookup using precomputed data structures."""
        building = class_info['building']
        
        # Get suitable room types for subject
        suitable_rooms = []
        room_requirements = self.config.get('room_requirements', {})
        
        if subject in room_requirements:
            requirements = room_requirements[subject]
            for room_type, ratio in requirements.items():
                if ratio > 0 and room_type.endswith('_ratio'):
                    room_type_name = room_type.replace('_ratio', '').title()
                    if room_type_name == 'Classroom':
                        room_type_name = 'Classroom'
                    
                    rooms = self.rooms_by_building_type.get((building, room_type_name), [])
                    suitable_rooms.extend(rooms)
        
        # Add classrooms as default if no specific rooms found
        if not suitable_rooms:
            suitable_rooms = self.rooms_by_building_type.get((building, 'Classroom'), [])
        
        return suitable_rooms

    def _is_room_suitable_for_subject(self, room: pd.Series, subject: str) -> bool:
        """Check if a room is suitable for a specific subject (cached version)."""
        room_requirements = self.config.get('room_requirements', {})
        
        if subject in room_requirements:
            requirements = room_requirements[subject]
            room_type = room['room_type']
            
            type_mapping = {
                'Lab': 'lab_ratio',
                'Playground': 'playground_ratio', 
                'Library': 'library_ratio',
                'Music': 'music_ratio',
                'Computer': 'computer_ratio',
                'Classroom': 'classroom_ratio'
            }
            
            ratio_key = type_mapping.get(room_type)
            if ratio_key and requirements.get(ratio_key, 0) > 0:
                return True
        
        # Default to classroom
        return room['room_type'] == 'Classroom'

    def is_slot_available_fast(self, teacher_id: str, room_id: str, day: str, slot: int) -> bool:
        """Fast slot availability check using set-based lookups."""
        time_slot = (day, slot)
        
        # Check teacher availability
        if teacher_id and time_slot in self.teacher_schedule[teacher_id]:
            return False
        
        # Check room availability
        if room_id and time_slot in self.room_usage[room_id]:
            return False
        
        return True

    def predict_teacher_assignment_score_batch(self, teachers: List, class_info: pd.Series, subject: str) -> List[float]:
        """Batch prediction for teacher assignment scores for better performance."""
        if not teachers:
            return []
        
        # Map class gender
        class_gender = class_info['gender']
        if class_gender == 'Boys':
            teacher_gender = 'Male'
        elif class_gender == 'Girls':
            teacher_gender = 'Female'
        else:
            teacher_gender = class_gender
        
        # Prepare features for all teachers at once
        features_batch = []
        for teacher in teachers:
            combination_score = self._get_subject_combination_score(
                str(teacher['primary_subject']), str(subject)
            )
            
            # Get teacher availability (use cached lookup)
            teacher_avail = np.mean([
                self.teacher_availability[teacher['teacher_id']][day][slot]
                for day in ['Sun', 'Mon', 'Tue', 'Wed', 'Thu']
                for slot in range(1, 9)
            ])
            
            features = [
                int(teacher['gender'] == teacher_gender),
                int(teacher['home_building'] == class_info['building']),
                int(teacher['primary_subject'] == subject),
                int(subject in (teacher['secondary_subjects'].split(',') if pd.notna(teacher['secondary_subjects']) else [])),
                int(teacher['primary_subject'] == subject or subject in (teacher['secondary_subjects'].split(',') if pd.notna(teacher['secondary_subjects']) else [])),
                float(combination_score),
                teacher_avail,
                int(teacher['daily_class_required'])
            ]
            features_batch.append(features)
        
        # Batch predict
        try:
            probas = self.models['teacher_assignment'].predict_proba(np.array(features_batch))
            if probas.shape[1] > 1:
                scores = probas[:, 1]
            else:
                scores = probas[:, 0]
        except:
            scores = np.full(len(teachers), 0.5)
        
        return scores.tolist()

    def _get_subject_combination_score(self, primary_subject: str, secondary_subject: str) -> float:
        """Get compatibility score between two subjects (cached lookups)."""
        # Use a simple caching approach
        cache_key = (primary_subject, secondary_subject)
        if hasattr(self, '_combination_cache'):
            if cache_key in self._combination_cache:
                return self._combination_cache[cache_key]
        else:
            self._combination_cache = {}
        
        # Calculate score
        for combo_name, combo_data in self.subject_combinations.items():
            if (combo_data.get('primary_subject') == primary_subject and 
                combo_data.get('secondary_subject') == secondary_subject):
                score = combo_data.get('compatibility_score', 0.5)
                self._combination_cache[cache_key] = score
                return score
            
            # Check reverse combination
            if (combo_data.get('primary_subject') == secondary_subject and 
                combo_data.get('secondary_subject') == primary_subject):
                score = combo_data.get('compatibility_score', 0.5) * 0.8
                self._combination_cache[cache_key] = score
                return score
        
        # Default score
        score = 0.3
        self._combination_cache[cache_key] = score
        return score

    def generate_schedule_for_class_optimized(self, class_info: pd.Series) -> List[ScheduleEntry]:
        """Optimized schedule generation for a specific class."""
        class_schedule = []
        class_id = class_info['class_id']
        
        # Get curriculum for this class (from precomputed cache)
        class_curriculum = self.curriculum_by_class[class_id]
        
        days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu']
        
        # Sort subjects by priority (based on grade level)
        grade = class_info['grade']
        if grade >= 7:
            sequence_type = 'high'
        elif grade >= 4:
            sequence_type = 'middle'
        else:
            sequence_type = 'elementary'
        
        subject_priorities = []
        for subject, periods_needed in class_curriculum.items():
            sequence_info = self.subject_sequences[sequence_type].get(subject, {'priority': 5})
            priority = sequence_info['priority']
            subject_priorities.append((priority, subject, periods_needed))
        
        # Sort by priority (lower priority number = higher importance)
        subject_priorities.sort(key=lambda x: x[0])
        
        for priority, subject, periods_needed in subject_priorities:
            periods_assigned = 0
            
            # Get available teachers and rooms using optimized lookup
            available_teachers = self.get_available_teachers_fast(class_info, subject)
            available_rooms = self.get_available_rooms_fast(class_info, subject)
            
            if not available_teachers or not available_rooms:
                print(f"    WARNING: No available teachers or rooms for {class_id} - {subject}")
                continue
            
            # Batch predict teacher scores for better performance
            teacher_scores = self.predict_teacher_assignment_score_batch(available_teachers, class_info, subject)
            teacher_score_pairs = list(zip(available_teachers, teacher_scores))
            
            # Sort by score and load balance
            teacher_score_pairs.sort(key=lambda x: (-x[1], x[0].get('current_load', 0)))
            
            # Get subject sequence for optimized slot selection
            sequence_info = self.subject_sequences[sequence_type].get(subject, {'suggested_slots': list(range(1, 9))})
            suggested_slots = sequence_info['suggested_slots']
            
            # Try to assign periods efficiently
            assigned_periods = self._assign_periods_optimized(
                teacher_score_pairs, class_info, subject, suggested_slots, 
                available_rooms, periods_needed, days
            )
            
            periods_assigned += assigned_periods
            
            if periods_assigned < periods_needed:
                print(f"    WARNING: Only assigned {periods_assigned}/{periods_needed} periods for {class_id} - {subject}")
        
        return class_schedule

    def _assign_periods_optimized(self, teacher_score_pairs: List, class_info: pd.Series, 
                                 subject: str, suggested_slots: List, available_rooms: List, 
                                 periods_needed: int, days: List) -> int:
        """Optimized period assignment algorithm."""
        periods_assigned = 0
        max_attempts = periods_needed * 10  # Reduced from 20 for speed
        attempts = 0
        
        while periods_assigned < periods_needed and attempts < max_attempts:
            attempts += 1
            assigned_this_round = False
            
            # Try teachers in priority order
            for teacher, teacher_score in teacher_score_pairs:
                if periods_assigned >= periods_needed:
                    break
                
                teacher_id = teacher['teacher_id']
                current_load = self.teacher_loads[teacher_id]
                
                # Check capacity
                if current_load >= teacher['max_periods']:
                    continue
                
                # Try suggested slots first for better scheduling
                for slot in suggested_slots:
                    if periods_assigned >= periods_needed:
                        break
                        
                    for day in days:
                        if periods_assigned >= periods_needed:
                            break
                        
                        # Fast availability check
                        if not self.is_slot_available_fast(teacher_id, '', day, slot):
                            continue
                        
                        # Check teacher availability (fast lookup)
                        if not self.teacher_availability[teacher_id][day][slot]:
                            continue
                        
                        # Find suitable room (optimized)
                        suitable_room = self._find_suitable_room_fast(available_rooms, day, slot, subject)
                        if suitable_room is None:
                            continue
                        
                        # Create assignment
                        entry = ScheduleEntry(
                            class_id=class_info['class_id'],
                            teacher_id=teacher_id,
                            subject=subject,
                            room_id=suitable_room['room_id'],
                            day=day,
                            slot=slot,
                            confidence=teacher_score
                        )
                        
                        # Update tracking structures
                        self.schedule.append(entry)
                        self.teacher_loads[teacher_id] += 1
                        self.teacher_schedule[teacher_id].add((day, slot))
                        self.room_usage[suitable_room['room_id']].add((day, slot))
                        
                        periods_assigned += 1
                        assigned_this_round = True
                        
                        # Update teacher's current load for next iterations
                        teacher['current_load'] = self.teacher_loads[teacher_id]
                        break
                    
                    if assigned_this_round:
                        break
                
                if assigned_this_round:
                    break
            
            # Exit if no progress made
            if not assigned_this_round:
                break
        
        return periods_assigned

    def _find_suitable_room_fast(self, available_rooms: List, day: str, slot: int, subject: str) -> Optional[pd.Series]:
        """Fast room finding with cached suitability checks."""
        for room in available_rooms:
            room_id = room['room_id']
            
            # Fast availability check
            if not self.is_slot_available_fast('', room_id, day, slot):
                continue
            
            # Use cached suitability check
            if self.room_subject_cache.get((room_id, subject), False):
                return room
        
        # Fallback to any available room
        for room in available_rooms:
            room_id = room['room_id']
            if not self.is_slot_available_fast('', room_id, day, slot):
                continue
            return room
        
        return None

    def generate_complete_schedule_optimized(self) -> List[ScheduleEntry]:
        """Generate complete schedule with grade-based prioritization and iterative improvements."""
        print("Generating optimized schedule with grade-based prioritization and iterative improvements...")
        
        # Grade-based prioritization: Process higher grades first for better resource allocation
        grade_priorities = {
            'g7_9': 1,  # Highest priority - High school
            'g4_6': 2,  # Medium priority - Middle school  
            'g1_3': 3   # Lower priority - Elementary
        }
        
        # Add grade category to classes for sorting
        classes_with_grade_category = self.classes_df.copy()
        classes_with_grade_category['grade_category'] = classes_with_grade_category['grade'].apply(
            lambda g: 'g7_9' if g >= 7 else ('g4_6' if g >= 4 else 'g1_3')
        )
        classes_with_grade_category['grade_priority'] = classes_with_grade_category['grade_category'].map(grade_priorities)
        
        # Sort by grade priority, then by grade, then by building
        sorted_classes = classes_with_grade_category.sort_values(
            ['grade_priority', 'grade', 'building'], 
            ascending=[True, False, True]
        )
        
        total_classes = len(sorted_classes)
        grade_stats = {}
        
        print(f"\nGrade-based Processing Order:")
        for grade_cat in ['g7_9', 'g4_6', 'g1_3']:
            grade_classes = sorted_classes[sorted_classes['grade_category'] == grade_cat]
            print(f"   {grade_cat.upper()}: {len(grade_classes)} classes")
        
        # Process classes by grade groups with iterative improvements
        for grade_cat in ['g7_9', 'g4_6', 'g1_3']:
            grade_classes = sorted_classes[sorted_classes['grade_category'] == grade_cat]
            if len(grade_classes) == 0:
                continue
                
            print(f"\nProcessing {grade_cat.upper()} ({len(grade_classes)} classes)")
            
            # Iterative improvement: Multiple passes for better optimization
            max_iterations = 3
            for iteration in range(max_iterations):
                print(f"   Iteration {iteration + 1}/{max_iterations}")
                
                for idx, (_, class_info) in enumerate(grade_classes.iterrows()):
                    class_id = class_info['class_id']
                    
                    # Skip if already fully scheduled
                    if class_id in self.class_schedules:
                        existing_periods = len(self.class_schedules[class_id])
                        required_periods = self.curriculum_df[
                            self.curriculum_df['class_id'] == class_id
                        ]['periods_per_week'].sum()
                        
                        if existing_periods >= required_periods * 0.95:  # 95% coverage threshold
                            continue
                    
                    class_schedule = self.generate_schedule_for_class_optimized(class_info)
                    self.class_schedules[class_id] = class_schedule
                
                # Calculate grade statistics
                grade_coverage = self._calculate_grade_coverage(grade_classes)
                grade_stats[grade_cat] = grade_coverage
                
                print(f"   Grade {grade_cat} coverage: {grade_coverage['coverage_rate']:.1f}%")
                
                # Early exit if coverage is good enough
                if grade_coverage['coverage_rate'] >= 90:
                    print(f"   SUCCESS: Grade {grade_cat} coverage target reached!")
                    break
        
        # Final iterative improvement pass across all grades
        print(f"\nFinal iterative improvement pass...")
        self._iterative_improvement_pass()
        
        print(f"\nSUCCESS: Grade-based schedule generation completed!")
        print(f"   - Total schedule entries: {len(self.schedule)}")
        print(f"   - Classes processed: {len(self.class_schedules)}")
        
        # Print grade statistics
        print(f"\nGrade Coverage Statistics:")
        for grade_cat, stats in grade_stats.items():
            print(f"   {grade_cat.upper()}: {stats['coverage_rate']:.1f}% coverage "
                  f"({stats['covered_periods']}/{stats['required_periods']} periods)")
        
        return self.schedule

    def _calculate_grade_coverage(self, grade_classes: pd.DataFrame) -> Dict:
        """Calculate coverage statistics for a grade group."""
        total_required = 0
        total_covered = 0
        
        for _, class_info in grade_classes.iterrows():
            class_id = class_info['class_id']
            
            # Get required periods for this class
            class_curriculum = self.curriculum_df[self.curriculum_df['class_id'] == class_id]
            required_periods = class_curriculum['periods_per_week'].sum()
            total_required += required_periods
            
            # Count covered periods
            if class_id in self.class_schedules:
                covered_periods = len(self.class_schedules[class_id])
                total_covered += covered_periods
        
        coverage_rate = (total_covered / total_required * 100) if total_required > 0 else 0
        
        return {
            'required_periods': total_required,
            'covered_periods': total_covered,
            'coverage_rate': coverage_rate
        }

    def _iterative_improvement_pass(self):
        """Perform iterative improvements across all classes."""
        print("   Performing iterative improvements...")
        
        # Identify classes with low coverage
        low_coverage_classes = []
        for class_id, class_schedule in self.class_schedules.items():
            class_info = self.classes_df[self.classes_df['class_id'] == class_id].iloc[0]
            class_curriculum = self.curriculum_df[self.curriculum_df['class_id'] == class_id]
            required_periods = class_curriculum['periods_per_week'].sum()
            covered_periods = len(class_schedule)
            
            coverage_rate = (covered_periods / required_periods * 100) if required_periods > 0 else 0
            
            if coverage_rate < 85:  # Less than 85% coverage
                low_coverage_classes.append((class_info, coverage_rate))
        
        # Sort by coverage rate (lowest first)
        low_coverage_classes.sort(key=lambda x: x[1])
        
        print(f"   Found {len(low_coverage_classes)} classes with low coverage")
        
        # Try to improve low coverage classes
        improvements_made = 0
        for class_info, current_coverage in low_coverage_classes[:10]:  # Limit to top 10 worst
            class_id = class_info['class_id']
            
            # Try to find additional assignments
            additional_periods = self._find_additional_assignments_for_class(class_info)
            
            if additional_periods > 0:
                improvements_made += 1
                print(f"   SUCCESS: Improved {class_id}: +{additional_periods} periods")
        
        print(f"   Iterative improvements: {improvements_made} classes enhanced")

    def _find_additional_assignments_for_class(self, class_info: pd.Series) -> int:
        """Find additional assignments for a class with low coverage."""
        class_id = class_info['class_id']
        periods_found = 0
        
        # Get curriculum requirements
        class_curriculum = self.curriculum_df[self.curriculum_df['class_id'] == class_id]
        
        for _, subject_row in class_curriculum.iterrows():
            subject = subject_row['subject']
            required_periods = subject_row['periods_per_week']
            
            # Count already assigned periods for this subject
            assigned_periods = len([
                entry for entry in self.schedule 
                if entry.class_id == class_id and entry.subject == subject
            ])
            
            if assigned_periods < required_periods:
                # Try to assign additional periods
                additional_needed = required_periods - assigned_periods
                
                # Get available teachers and rooms
                available_teachers = self.get_available_teachers(class_info, subject)
                available_rooms = self.get_available_rooms(class_info, subject)
                
                if available_teachers and available_rooms:
                    # Try to assign remaining periods
                    periods_assigned = self._assign_remaining_periods(
                        class_info, subject, available_teachers, available_rooms, additional_needed
                    )
                    periods_found += periods_assigned
        
        return periods_found

    def _assign_remaining_periods(self, class_info: pd.Series, subject: str, 
                                 available_teachers: List, available_rooms: List, 
                                 periods_needed: int) -> int:
        """Assign remaining periods for a specific class-subject combination."""
        periods_assigned = 0
        days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu']
        
        for teacher in available_teachers:
            if periods_assigned >= periods_needed:
                break
                
            teacher_id = teacher['teacher_id']
            current_load = self.teacher_loads.get(teacher_id, 0)
            
            if current_load >= teacher['max_periods']:
                continue
            
            # Try to assign periods
            for day in days:
                if periods_assigned >= periods_needed:
                    break
                    
                for slot in range(1, 9):
                    if periods_assigned >= periods_needed:
                        break
                    
                    # Check availability
                    if not self.is_slot_available_fast(teacher_id, '', day, slot):
                        continue
                    
                    if not self.teacher_availability[teacher_id][day][slot]:
                        continue
                    
                    # Find suitable room
                    suitable_room = self._find_suitable_room_fast(available_rooms, day, slot, subject)
                    if suitable_room is None:
                        continue
                    
                    # Create assignment
                    entry = ScheduleEntry(
                        class_id=class_info['class_id'],
                        teacher_id=teacher_id,
                        subject=subject,
                        room_id=suitable_room['room_id'],
                        day=day,
                        slot=slot,
                        confidence=0.8  # Good confidence for iterative improvement
                    )
                    
                    # Update tracking structures
                    self.schedule.append(entry)
                    self.teacher_loads[teacher_id] += 1
                    self.teacher_schedule[teacher_id].add((day, slot))
                    self.room_usage[suitable_room['room_id']].add((day, slot))
                    
                    periods_assigned += 1
        
        return periods_assigned

    def save_schedule(self, schedule: List[ScheduleEntry]):
        """Save generated schedule to CSV file."""
        schedule_data = []
        for entry in schedule:
            schedule_data.append({
                'class_id': entry.class_id,
                'teacher_id': entry.teacher_id,
                'subject': entry.subject,
                'room_id': entry.room_id,
                'day': entry.day,
                'slot': entry.slot,
                'confidence': entry.confidence
            })
        
        schedule_df = pd.DataFrame(schedule_data)
        schedule_df.to_csv(self.config['paths']['output_files']['schedule'], index=False)
        
        print(f"Schedule saved to: {self.config['paths']['output_files']['schedule']}")

    def print_schedule_summary(self, schedule: List[ScheduleEntry]):
        """Print summary of generated schedule."""
        print("\nOptimized Schedule Generation Summary")
        print("=" * 50)
        
        # Count by subject
        subject_counts = Counter(entry.subject for entry in schedule)
        
        print("\nSubject Distribution:")
        for subject, count in subject_counts.most_common():
            print(f"  {subject}: {count} periods")
        
        # Teacher load statistics
        teacher_counts = Counter(entry.teacher_id for entry in schedule)
        
        print(f"\nTeacher Load Distribution:")
        print(f"  Total teachers used: {len(teacher_counts)}")
        if teacher_counts:
            loads = list(teacher_counts.values())
            print(f"  Average load: {np.mean(loads):.1f} periods")
            print(f"  Min load: {min(loads)} periods")
            print(f"  Max load: {max(loads)} periods")
        
        # Room utilization
        room_counts = Counter(entry.room_id for entry in schedule)
        print(f"\nRoom Utilization:")
        print(f"  Total rooms used: {len(room_counts)}")
        if room_counts:
            print(f"  Average utilization: {np.mean(list(room_counts.values())):.1f} periods")
        
        # Confidence scores
        confidences = [entry.confidence for entry in schedule]
        if confidences:
            print(f"\nConfidence Scores:")
            print(f"  Average confidence: {np.mean(confidences):.4f}")
            print(f"  Min confidence: {min(confidences):.4f}")
            print(f"  Max confidence: {max(confidences):.4f}")

def main():
    """Main function for optimized schedule generation."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'system_config.yaml')
    
    print("Starting Step 4: Optimized Schedule Generation")
    print("=" * 50)
    
    # Initialize optimized schedule generator
    generator = OptimizedScheduleGenerator(config_path)
    
    # Generate complete schedule
    schedule = generator.generate_complete_schedule_optimized()
    
    # Save schedule
    generator.save_schedule(schedule)
    
    # Print summary
    generator.print_schedule_summary(schedule)
    
    print("\nStep 4 completed successfully with optimizations!")
    print("   Ready for Step 5: Validation")

if __name__ == "__main__":
    main()