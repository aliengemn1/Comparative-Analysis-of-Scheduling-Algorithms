"""
Step 2: Feature Engineering Module
===================================

This module extracts and engineers features for automated schedule generation:
- Teacher-class compatibility features
- Scheduling context features
- Constraint satisfaction features
- Quality prediction features

Usage:
    python 02_feature_engineering/main.py
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FeatureEngineer:
    def __init__(self, config_path: str):
        """Initialize feature engineer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)

        # Create output directory
        os.makedirs(self.config['paths']['features_dir'], exist_ok=True)

        # Load base data
        self.classes_df = pd.read_csv(self.config['paths']['input_files']['classes'])
        self.teachers_df = pd.read_csv(self.config['paths']['input_files']['teachers'])
        self.rooms_df = pd.read_csv(self.config['paths']['input_files']['rooms'])
        self.curriculum_df = pd.read_csv(self.config['paths']['input_files']['curriculum'])
        self.availability_df = pd.read_csv(self.config['paths']['input_files']['teacher_availability'])
        
        # Load subject combinations matrix
        self.subject_combinations = self.config.get('subject_combinations', {})

        # Normalize columns expected downstream
        for col in ["gender", "home_building", "primary_subject", "secondary_subjects"]:
            if col in self.teachers_df.columns:
                self.teachers_df[col] = self.teachers_df[col].fillna("")

        # Day name mapping for consistency
        self.day_mapping = {
            'Sunday': 'Sun', 'Monday': 'Mon', 'Tuesday': 'Tue',
            'Wednesday': 'Wed', 'Thursday': 'Thu'
        }

        # Normalize availability types
        if "available" in self.availability_df.columns:
            # Accept truthy values like {0/1, "Y"/"N", True/False}
            self.availability_df["available"] = (
                self.availability_df["available"]
                .apply(lambda v: 1 if str(v).strip().lower() in {"1", "true", "t", "y", "yes"} else 0)
                .astype(int)
            )

        # Ensure numeric teacher limits
        for col in ["daily_class_required", "min_periods", "max_periods"]:
            if col in self.teachers_df.columns:
                self.teachers_df[col] = pd.to_numeric(self.teachers_df[col], errors="coerce").fillna(0).astype(int)

        # Ensure numeric curriculum periods
        if "periods_per_week" in self.curriculum_df.columns:
            self.curriculum_df["periods_per_week"] = pd.to_numeric(
                self.curriculum_df["periods_per_week"], errors="coerce"
            ).fillna(0).astype(int)

        if "periods_per_week" in self.classes_df.columns:
            self.classes_df["periods_per_week"] = pd.to_numeric(
                self.classes_df["periods_per_week"], errors="coerce"
            ).fillna(0).astype(int)

    def _map_teacher_to_class_gender(self, teacher_gender: str) -> str:
        """Map teacher gender ('Male'/'Female') to class gender label ('Boys'/'Girls')."""
        return "Boys" if str(teacher_gender).strip().lower() == "male" else "Girls"

    def _get_teacher_availability(self, teacher_id: str, day: str, slot: int) -> bool:
        """Safely get teacher availability for a specific day/slot."""
        day_abbrev = self.day_mapping.get(day, day)
        df = self.availability_df
        avail = df[
            (df.get('teacher_id') == teacher_id) &
            (df.get('day') == day_abbrev) &
            (df.get('slot') == slot)
        ]["available"]
        if len(avail) == 0:
            return False
        return bool(int(avail.iloc[0]))

    def _avg_teacher_availability(self, teacher_id: str) -> float:
        """Average availability across all recorded time slots for a teacher."""
        df = self.availability_df
        avail = df[df.get('teacher_id') == teacher_id]["available"]
        if len(avail) == 0:
            return 0.0
        return float(pd.to_numeric(avail, errors="coerce").fillna(0).astype(int).mean())

    def _get_room_type_for_subject(self, subject: str) -> str:
        """Determine required room type for a subject."""
        subject_lower = str(subject).strip().lower()
        if subject_lower in {'science', 'computer', 'physics', 'chemistry', 'biology', 'ict'}:
            return 'Lab'
        elif subject_lower in {'pe', 'physical education', 'sports'}:
            return 'Playground'
        elif subject_lower in {'library', 'reading'}:
            return 'Library'
        else:
            return 'Classroom'

    def _get_subject_combination_score(self, primary_subject: str, secondary_subject: str) -> float:
        """Get compatibility score between two subjects based on combinations matrix."""
        # Look for direct combination
        for combo_name, combo_data in self.subject_combinations.items():
            if (combo_data.get('primary_subject') == primary_subject and 
                combo_data.get('secondary_subject') == secondary_subject):
                return combo_data.get('compatibility_score', 0.5)
            
            # Check reverse combination
            if (combo_data.get('primary_subject') == secondary_subject and 
                combo_data.get('secondary_subject') == primary_subject):
                return combo_data.get('compatibility_score', 0.5) * 0.8  # Slightly lower for reverse
        
        # Default compatibility score for non-defined combinations
        return 0.3

    def extract_teacher_class_compatibility_features(self) -> pd.DataFrame:
        """Extract features for teacher-class-subject compatibility."""
        rows: List[Dict[str, Any]] = []

        for _, teacher in self.teachers_df.iterrows():
            teacher_gender_mapped = self._map_teacher_to_class_gender(teacher.get('gender', ''))
            # Normalize secondary subjects list
            sec = (teacher.get('secondary_subjects') or '')
            secondary_subjects = [s.strip() for s in str(sec).split(',') if s.strip()]

            # Pre-compute availability mean once
            teacher_availability = self._avg_teacher_availability(teacher.get('teacher_id'))

            for _, class_info in self.classes_df.iterrows():
                gender_compatible = teacher_gender_mapped == class_info.get('gender')
                building_match = str(teacher.get('home_building')) == str(class_info.get('building'))

                class_subjects = self.curriculum_df[
                    self.curriculum_df['class_id'] == class_info['class_id']
                ]['subject'].tolist()

                for subject in class_subjects:
                    primary_subject_match = str(teacher.get('primary_subject')) == str(subject)
                    secondary_subject_capability = str(subject) in secondary_subjects
                    subject_capability = primary_subject_match or secondary_subject_capability
                    
                    # Calculate subject combination compatibility score
                    combination_score = self._get_subject_combination_score(
                        str(teacher.get('primary_subject')), str(subject)
                    )

                    # Calculate grade-based features
                    grade = class_info.get('grade', 1)
                    grade_category = 'g7_9' if grade >= 7 else ('g4_6' if grade >= 4 else 'g1_3')
                    grade_priority = 1 if grade >= 7 else (2 if grade >= 4 else 3)
                    
                    rows.append({
                        'teacher_id': teacher.get('teacher_id'),
                        'class_id': class_info.get('class_id'),
                        'subject': subject,
                        'grade': int(grade),
                        'grade_category': grade_category,
                        'grade_priority': int(grade_priority),
                        'gender_compatible': int(bool(gender_compatible)),
                        'building_match': int(bool(building_match)),
                        'primary_subject_match': int(bool(primary_subject_match)),
                        'secondary_subject_capability': int(bool(secondary_subject_capability)),
                        'subject_capability': int(bool(subject_capability)),
                        'subject_combination_score': float(combination_score),
                        'teacher_availability': float(teacher_availability),
                        'daily_class_required': int(teacher.get('daily_class_required', 0)),
                        'teacher_min_periods': int(teacher.get('min_periods', 0)),
                        'teacher_max_periods': int(teacher.get('max_periods', 0)),
                        'class_periods_per_week': int(class_info.get('periods_per_week', 0))
                    })

        return pd.DataFrame(rows)

    def extract_scheduling_context_features(self) -> pd.DataFrame:
        """Extract features for scheduling context."""
        rows: List[Dict[str, Any]] = []
        days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']
        slots = list(range(1, int(self.config['schedule']['slots_per_day']) + 1))

        for _, class_info in self.classes_df.iterrows():
            cls_id = class_info['class_id']
            building = class_info.get('building')
            grade = class_info.get('grade')

            class_subjects = self.curriculum_df[self.curriculum_df['class_id'] == cls_id]

            # Calculate specialized room requirements
            subjects = class_subjects['subject']
            periods = class_subjects['periods_per_week']
            
            lab_mask = subjects.apply(self._get_room_type_for_subject).eq('Lab')
            playground_mask = subjects.apply(self._get_room_type_for_subject).eq('Playground')
            library_mask = subjects.apply(self._get_room_type_for_subject).eq('Library')
            music_mask = subjects.apply(self._get_room_type_for_subject).eq('Music')
            computer_mask = subjects.apply(self._get_room_type_for_subject).eq('Computer')
            lab_required = int((lab_mask * periods).sum())
            playground_required = int((playground_mask * periods).sum())
            library_required = int((library_mask * periods).sum())
            music_required = int((music_mask * periods).sum())
            computer_required = int((computer_mask * periods).sum())
            classroom_required = int(periods.sum() - lab_required - playground_required - library_required - music_required - computer_required)

            building_rooms = self.rooms_df[self.rooms_df['building'] == building]
            available_classrooms = int((building_rooms['room_type'] == 'Classroom').sum())
            available_labs = int((building_rooms['room_type'] == 'Lab').sum())
            available_playgrounds = int((building_rooms['room_type'] == 'Playground').sum())
            available_libraries = int((building_rooms['room_type'] == 'Library').sum())
            available_music = int((building_rooms['room_type'] == 'Music').sum())
            available_computers = int((building_rooms['room_type'] == 'Computer').sum())
            available_music = int((building_rooms['room_type'] == 'Music').sum())
            available_computers = int((building_rooms['room_type'] == 'Computer').sum())
            total_special_rooms = available_labs + available_playgrounds + available_libraries + available_music + available_computers
            total_special_requirements = lab_required + playground_required + library_required + music_required + computer_required
            room_utilization_pressure = float(total_special_requirements) / max(1, total_special_rooms)

            for day in days:
                for slot in slots:
                    rows.append({
                        'class_id': cls_id,
                        'day': day,
                        'slot': int(slot),
                        'building': building,
                        'grade': grade,
                        'lab_required': int(lab_required),
                        'playground_required': int(playground_required),
                        'music_required': int(music_required),
                        'computer_required': int(computer_required),
                        'library_required': int(library_required),
                        'classroom_required': int(classroom_required),
                        'available_classrooms': int(available_classrooms),
                        'available_labs': int(available_labs),
                        'available_music': int(available_music),
                        'available_computers': int(available_computers),
                        'available_playgrounds': int(available_playgrounds),
                        'available_libraries': int(available_libraries),
                        'available_music': int(available_music),
                        'available_computers': int(available_computers),
                        'room_utilization_pressure': float(room_utilization_pressure)
                    })

        return pd.DataFrame(rows)

    def extract_constraint_satisfaction_features(self) -> pd.DataFrame:
        """Extract features for constraint satisfaction prediction."""
        rows: List[Dict[str, Any]] = []
        days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']
        slots_per_day = int(self.config['schedule']['slots_per_day'])

        n_samples = int(self.config.get('feature_sampling', {}).get('constraint_samples', 1000))
        for _ in range(n_samples):
            teacher = self.teachers_df.sample(1).iloc[0]
            class_info = self.classes_df.sample(1).iloc[0]
            cls_id = class_info['class_id']
            subjects = self.curriculum_df[self.curriculum_df['class_id'] == cls_id]['subject']
            if len(subjects) == 0:
                continue
            subject = subjects.sample(1).iloc[0]

            # Random room in same building (fallback to any room if none)
            available_rooms = self.rooms_df[self.rooms_df['building'] == class_info['building']]
            room = (available_rooms if len(available_rooms) else self.rooms_df).sample(1).iloc[0]

            day = random.choice(days)
            slot = random.randint(1, slots_per_day)

            teacher_gender_mapped = self._map_teacher_to_class_gender(teacher.get('gender', ''))
            gender_compatible = (teacher_gender_mapped == class_info.get('gender'))
            teacher_available = self._get_teacher_availability(teacher.get('teacher_id'), day, slot)

            required_room_type = self._get_room_type_for_subject(subject)
            room_type_compatible = (str(room.get('room_type')) == required_room_type)

            feasible = bool(gender_compatible and teacher_available and room_type_compatible)

            rows.append({
                'teacher_id': teacher.get('teacher_id'),
                'class_id': cls_id,
                'subject': subject,
                'room_id': room.get('room_id'),
                'day': day,
                'slot': int(slot),
                'gender_compatible': int(bool(gender_compatible)),
                'teacher_available': int(bool(teacher_available)),
                'room_type_compatible': int(bool(room_type_compatible)),
                'building_match': int(str(teacher.get('home_building')) == str(class_info.get('building'))),
                'primary_subject_match': int(str(teacher.get('primary_subject')) == str(subject)),
                'feasible': int(bool(feasible))
            })

        return pd.DataFrame(rows)

    def extract_quality_prediction_features(self) -> pd.DataFrame:
        """Extract features for schedule quality prediction."""
        rows: List[Dict[str, Any]] = []
        n_samples = int(self.config.get('feature_sampling', {}).get('quality_samples', 500))

        for _ in range(n_samples):
            class_info = self.classes_df.sample(1).iloc[0]
            cls_id = class_info['class_id']

            class_subjects = self.curriculum_df[self.curriculum_df['class_id'] == cls_id]
            total_periods = int(class_subjects['periods_per_week'].sum())
            unique_subjects = int(class_subjects['subject'].nunique())

            # Filter teachers by class gender using the same mapping
            available_teachers = self.teachers_df[
                self.teachers_df['gender'].apply(self._map_teacher_to_class_gender) == class_info.get('gender')
            ]
            building_rooms = self.rooms_df[self.rooms_df['building'] == class_info.get('building')]

            teacher_utilization = min(1.0, (total_periods / max(1, len(available_teachers) * 20))) if len(available_teachers) > 0 else 0.0
            room_utilization = min(1.0, (total_periods / max(1, len(building_rooms) * 35))) if len(building_rooms) > 0 else 0.0

            # Schedule balance (variance in daily periods)
            # Use config slots_per_day to keep scale realistic
            slots_per_day = int(self.config['schedule']['slots_per_day'])
            # pseudo schedule to compute balance
            daily_periods = [random.randint(max(1, slots_per_day - 2), slots_per_day) for _ in range(5)]
            mean_dp = float(np.mean(daily_periods))
            var_dp = float(np.var(daily_periods))
            schedule_balance = float(1.0 - (var_dp / max(1.0, mean_dp)))

            quality_score = float(0.3 * teacher_utilization + 0.3 * room_utilization + 0.4 * schedule_balance)

            rows.append({
                'class_id': cls_id,
                'grade': class_info.get('grade'),
                'building': class_info.get('building'),
                'total_periods': int(total_periods),
                'unique_subjects': int(unique_subjects),
                'available_teachers': int(len(available_teachers)),
                'available_rooms': int(len(building_rooms)),
                'teacher_utilization': float(teacher_utilization),
                'room_utilization': float(room_utilization),
                'schedule_balance': float(schedule_balance),
                'quality_score': float(quality_score)
            })

        return pd.DataFrame(rows)

    def save_features(self,
                      compatibility_features: pd.DataFrame,
                      context_features: pd.DataFrame,
                      constraint_features: pd.DataFrame,
                      quality_features: pd.DataFrame):
        """Save all extracted features to CSV files."""
        out_dir = self.config['paths']['features_dir']

        # Ensure all numeric dtypes where appropriate (prevents CSV dtype surprises)
        for df in [compatibility_features, context_features, constraint_features, quality_features]:
            for col in df.columns:
                if df[col].dtype == "boolean":
                    df[col] = df[col].astype(int)

        paths = {
            "teacher_class_compatibility.csv": compatibility_features,
            "scheduling_context.csv": context_features,
            "constraint_satisfaction.csv": constraint_features,
            "quality_prediction.csv": quality_features
        }

        for fname, df in paths.items():
            df.to_csv(os.path.join(out_dir, fname), index=False)

        print("Feature engineering completed!")
        print(f"   - Teacher-class compatibility features: {len(compatibility_features)}")
        print(f"   - Scheduling context features: {len(context_features)}")
        print(f"   - Constraint satisfaction features: {len(constraint_features)}")
        print(f"   - Quality prediction features: {len(quality_features)}")
        print(f"   - Features saved to: {out_dir}")

def main():
    """Main function for feature engineering."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'system_config.yaml')

    print("Starting Step 2: Feature Engineering")
    print("=" * 50)

    try:
        engineer = FeatureEngineer(config_path)

        print("Extracting teacher-class compatibility features...")
        compatibility_features = engineer.extract_teacher_class_compatibility_features()

        print("Extracting scheduling context features...")
        context_features = engineer.extract_scheduling_context_features()

        print("Extracting constraint satisfaction features...")
        constraint_features = engineer.extract_constraint_satisfaction_features()

        print("Extracting quality prediction features...")
        quality_features = engineer.extract_quality_prediction_features()

        print("Saving features...")
        engineer.save_features(compatibility_features, context_features, constraint_features, quality_features)

        print("\nStep 2 completed successfully!")
        print("   Ready for Step 3: Model Training")

    except Exception as e:
        print(f"\nError in feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    main()
