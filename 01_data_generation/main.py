"""
Step 1: Data Generation Module
==============================

This module generates the base datasets needed for automated schedule generation:
- Classes, Teachers, Rooms
- Curriculum requirements
- Teacher availability
- Base scheduling data

Usage:
    python 01_data_generation/main.py
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ----------------------------
# Utilities
# ----------------------------

def unique_teacher_name(idx: int, gender: str, used: set) -> str:
    male_names = [
        # Egyptian
        "Ahmed", "Mohamed", "Mahmoud", "Youssef", "Mostafa", "Omar", "Ali", "Khaled", "Hassan", "Ibrahim",
        "Karim", "Tarek", "Amr", "Hany", "Ashraf", "Walid", "Alaa", "Ayman", "Sherif", "Nader",
        # Palestinian/Levant
        "Yazan", "Bashar", "Laith", "Murad", "Fares", "Ziad", "Nidal", "Samir", "Rami", "Nasser",
        "Adel", "Majed", "Fadi", "Karam", "Hamza", "Jamal", "Anas", "Othman", "Bilal", "Qasem",
        # Saudi
        "Abdullah", "Abdulaziz", "Abdulrahman", "Faisal", "Salman", "Turki", "Mishal", "Nawaf", "Talal", "Bandar",
        "Fahd", "Badr", "Rashed", "Hamad", "Mansour", "Sultan"
    ]
    female_names = [
        # Egyptian
        "Fatma", "Mariam", "Aya", "Salma", "Nour", "Heba", "Dina", "Yasmin", "Sara", "Hagar",
        "Laila", "Maha", "Noha", "Reem", "Doaa", "Asmaa", "Nadia", "Eman", "Hend", "Ghada",
        # Palestinian/Levant
        "Aisha", "Huda", "Rania", "Hanin", "Rula", "Dalia", "Lama", "Maysaa", "Suha", "Nahla",
        "Nisreen", "Mona", "Tamara", "Ruba", "Duaa", "Nadine", "Kholoud", "Samar", "Razan", "Jumana",
        # Saudi
        "Nouf", "Hessa", "Jawaher", "Noura", "Reema", "Abeer", "Lujain", "Rawan", "Shahad", "Ghadir", "Wijdan", "Dalal"
    ]
    last_names = [
        # Common Egyptian and Palestinian/Levant family names
        "El-Sayed", "Hassan", "Ibrahim", "Abdelrahman", "Mostafa", "Ashour", "Saad", "Shawky", "Gamal", "Fathy",
        "Al-Najjar", "Al-Tamimi", "Barghouti", "Khalidi", "Qudsi", "Al-Qaisi", "Al-Husseini", "Hamdan", "Zayed", "Salem",
        "Al-Ahmad", "Al-Rashid", "Al-Omari", "Al-Shammari", "Al-Qudah", "Al-Masri", "Al-Qudwa", "Al-Khatib", "Najm", "Sabri",
        "Shihab", "Abu Zaid", "Abu Hamid", "Abu Hassan", "Sabbagh", "Kassab", "Nasser", "Rahman", "Ismail", "Fahmy",
        # Saudi
        "Al-Saud", "Al-Faisal", "Al-Qahtani", "Al-Otaibi", "Al-Mutairi", "Al-Subaie", "Al-Dosari", "Al-Anazi", "Al-Zahrani", "Al-Ghamdi", "Al-Harbi"
    ]
    first_pool = male_names if gender == "Male" else female_names
    # Try deterministic spread first
    first = first_pool[idx % len(first_pool)]
    last = last_names[(idx * 7) % len(last_names)]
    candidate = f"{first} {last}"
    if candidate not in used:
        used.add(candidate)
        return candidate
    # Fallback: try another deterministic mix
    first2 = first_pool[(idx * 13) % len(first_pool)]
    last2 = last_names[(idx * 17 + 3) % len(last_names)]
    candidate2 = f"{first2} {last2}"
    if candidate2 not in used:
        used.add(candidate2)
        return candidate2
    # Last resort: append numeric suffix to ensure uniqueness
    suffix = 1
    while True:
        candidate3 = f"{first} {last} {suffix}"
        if candidate3 not in used:
            used.add(candidate3)
            return candidate3
        suffix += 1

# ----------------------------
# Data Classes
# ----------------------------

@dataclass
class ClassInfo:
    class_id: str
    grade: int
    section: str
    gender: str  # Boys/Girls
    building: str  # A/B
    periods_per_week: int

@dataclass
class Teacher:
    teacher_id: str
    name: str
    gender: str
    primary_subject: str
    secondary_subjects: List[str]
    home_building: str
    daily_class_required: bool
    min_periods: int
    max_periods: int

@dataclass
class Room:
    room_id: str
    room_type: str
    building: str
    capacity: int

# ----------------------------
# Generator
# ----------------------------

class DataGenerator:
    def __init__(self, config_path: str):
        """Initialize data generator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)

        # Create output directory
        os.makedirs(self.config['paths']['data_dir'], exist_ok=True)

        # Define specialized subjects (spelling must match curriculum keys)
        self.specialized_subjects = ['library', 'french', 'pe', 'art', 'computer','chinese','music']

        # Per-subject limits (prefer YAML override; fallback to your requested targets)
        default_limits = {
            'library': 6, 'french': 6, 'pe': 6, 'art': 6,
            'computer': 6, 'music': 6, 'chinese': 2
        }
        self.specialized_teachers_per_subject: Dict[str, int] = (
            self.config.get('teachers', {})
                .get('specialized_teachers_per_subject', default_limits)
        )

        # Load subject combinations matrix
        self.subject_combinations = self.config.get('subject_combinations', {})

        # Validate total_count (warn if too small)
        needed_specialized = sum(self.specialized_teachers_per_subject.get(s, 0) for s in self.specialized_subjects)
        self.total_teacher_budget = int(self.config['teachers']['total_count'])
        if self.total_teacher_budget < needed_specialized:
            # Auto-bump in-memory so generation doesn't fail; user can persist later in YAML if desired.
            print(f" [WARN] teachers.total_count ({self.total_teacher_budget}) < required specialized ({needed_specialized}). "
                  f"Temporarily increasing budget to {needed_specialized}.")
            self.total_teacher_budget = needed_specialized

    # --------- Classes ---------

    def generate_classes(self) -> List[ClassInfo]:
        """Generate all classes based on configuration"""
        classes: List[ClassInfo] = []

        # Grades 1-6: 20 sections each (10 boys + 10 girls)
        for grade in range(1, 7):
            for gender in ["Boys", "Girls"]:
                for s in range(1, 11):
                    section = f"{gender[0]}{s:02d}"
                    if grade <= 3:
                        building = "A"
                    else:
                        # split evenly across A/B per grade and gender
                        building = "A" if s <= 5 else "B"
                    classes.append(ClassInfo(
                        class_id=f"G{grade}-{section}",
                        grade=grade,
                        section=section,
                        gender=gender,
                        building=building,
                        periods_per_week=self.config['school']['grades']['g1_3']['periods_per_week'] if grade <= 3 else self.config['school']['grades']['g4_6']['periods_per_week']
                    ))

        # Grades 7-9: 12 sections each (6 boys + 6 girls) -> B
        for grade in range(7, 10):
            for gender in ["Boys", "Girls"]:
                for s in range(1, 7):
                    section = f"{gender[0]}{s:02d}"
                    classes.append(ClassInfo(
                        class_id=f"G{grade}-{section}",
                        grade=grade,
                        section=section,
                        gender=gender,
                        building="B",
                        periods_per_week=self.config['school']['grades']['g7_9']['periods_per_week']
                    ))

        return classes

    # --------- Teachers ---------

    def generate_teachers(self) -> List[Teacher]:
        """Generate teachers with specialized subject exact counts."""
        teachers: List[Teacher] = []
        teacher_config = self.config['teachers']
        subjects = list(self.config['curriculum']['subjects'].keys())

        # Which specialized subjects are actually needed? (check all grade ranges g4_6 and g7_9)
        specialized_teacher_counts: Dict[str, int] = {}
        for subject in self.specialized_subjects:
            if subject in subjects:
                subject_cfg = self.config['curriculum']['subjects'].get(subject, {})
                g1_3_periods = subject_cfg.get('g1_3', 0)
                g4_6_periods = subject_cfg.get('g4_6', 0)
                g7_9_periods = subject_cfg.get('g7_9', 0)
                if (g1_3_periods and g1_3_periods > 0) or (g4_6_periods and g4_6_periods > 0) or (g7_9_periods and g7_9_periods > 0):
                    specialized_teacher_counts[subject] = int(self.specialized_teachers_per_subject.get(subject, 0))
                else:
                    specialized_teacher_counts[subject] = 0
            else:
                specialized_teacher_counts[subject] = 0

        # Name uniqueness tracking & ID counter
        used_names: set = set()
        teacher_id_counter = 1

        def next_teacher_id() -> str:
            nonlocal teacher_id_counter
            tid = f"T{teacher_id_counter:03d}"
            teacher_id_counter += 1
            return tid

        # ===== Create specialized teachers: EXACT per-subject totals (no gender doubling) =====
        for subject, count in specialized_teacher_counts.items():
            for i in range(count):
                # alternate genders M/F/M/F...
                gender = 'Male' if (i % 2 == 0) else 'Female'
                name = unique_teacher_name(teacher_id_counter - 1, gender, used_names)

                # Secondary subjects using subject combinations matrix
                secondary_subjects = self._get_secondary_subjects_for_teacher(subject, subjects)

                if teacher_id_counter > self.total_teacher_budget:
                    break

                teachers.append(Teacher(
                    teacher_id=next_teacher_id(),
                    name=name,
                    gender=gender,
                    primary_subject=subject,
                    secondary_subjects=secondary_subjects,
                    home_building=random.choice(['A', 'B']),
                    daily_class_required=random.random() < teacher_config['daily_class_required_ratio'],
                    min_periods=teacher_config['min_periods_per_week'],
                    max_periods=teacher_config['max_periods_per_week']
                ))

                if teacher_id_counter > self.total_teacher_budget:
                    break

        # ===== Create general teachers (if budget remains) =====
        general_subjects = [s for s in subjects if s not in self.specialized_subjects]
        while teacher_id_counter <= self.total_teacher_budget:
            gender = random.choice(['Male', 'Female'])
            name = unique_teacher_name(teacher_id_counter - 1, gender, used_names)

            # Prefer general subjects; if empty, fall back to all subjects
            primary_subject = random.choice(general_subjects) if general_subjects else random.choice(subjects)

            # Secondary subjects using subject combinations matrix
            secondary_subjects = self._get_secondary_subjects_for_teacher(primary_subject, subjects)

            teachers.append(Teacher(
                teacher_id=next_teacher_id(),
                name=name,
                gender=gender,
                primary_subject=primary_subject,
                secondary_subjects=secondary_subjects,
                home_building=random.choice(['A', 'B']),
                daily_class_required=random.random() < teacher_config['daily_class_required_ratio'],
                min_periods=teacher_config['min_periods_per_week'],
                max_periods=teacher_config['max_periods_per_week']
            ))

        # Simple post-check: ensure exact specialized counts
        produced_counts: Dict[str, int] = {s: 0 for s in self.specialized_subjects}
        for t in teachers:
            if t.primary_subject in produced_counts:
                produced_counts[t.primary_subject] += 1

        for s, expected in specialized_teacher_counts.items():
            actual = produced_counts.get(s, 0)
            if actual != expected:
                print(f" [WARN] Specialized count mismatch for '{s}': expected {expected}, produced {actual}")

        return teachers

    def _get_secondary_subjects_for_teacher(self, primary_subject: str, all_subjects: List[str]) -> List[str]:
        """Get secondary subjects for a teacher based on subject combinations matrix."""
        secondary_subjects = []
        
        # Look for combinations where this subject is the primary
        for combo_name, combo_data in self.subject_combinations.items():
            if combo_data.get('primary_subject') == primary_subject:
                secondary_subject = combo_data.get('secondary_subject')
                if secondary_subject and secondary_subject in all_subjects:
                    # Add with probability based on compatibility score
                    compatibility_score = combo_data.get('compatibility_score', 0.5)
                    if random.random() < compatibility_score:
                        secondary_subjects.append(secondary_subject)
        
        # Also look for reverse combinations (where this subject could be secondary)
        for combo_name, combo_data in self.subject_combinations.items():
            if combo_data.get('secondary_subject') == primary_subject:
                primary_subject_candidate = combo_data.get('primary_subject')
                if primary_subject_candidate and primary_subject_candidate in all_subjects:
                    # Add with lower probability since it's reverse
                    compatibility_score = combo_data.get('compatibility_score', 0.5)
                    if random.random() < (compatibility_score * 0.7):  # Lower probability for reverse
                        secondary_subjects.append(primary_subject_candidate)
        
        # Limit to maximum 2 secondary subjects
        return secondary_subjects[:2]

    # --------- Rooms ---------

    def generate_rooms(self) -> List[Room]:
        """Generate all rooms based on configuration."""
        rooms: List[Room] = []
        room_config = self.config['rooms']

        # Classrooms in Building A
        for i in range(room_config['classroom_a']):
            rooms.append(Room(
                room_id=f"A-CR-{i+1:03d}",
                room_type="Classroom",
                building="A",
                capacity=30
            ))

        # Classrooms in Building B
        for i in range(room_config['classroom_b']):
            rooms.append(Room(
                room_id=f"B-CR-{i+1:03d}",
                room_type="Classroom",
                building="B",
                capacity=30
            ))

        # Labs in Building A
        for i in range(room_config['lab_a']):
            rooms.append(Room(
                room_id=f"A-LAB-{i+1:02d}",
                room_type="Lab",
                building="A",
                capacity=30
            ))

        # Labs in Building B
        for i in range(room_config['lab_b']):
            rooms.append(Room(
                room_id=f"B-LAB-{i+1:02d}",
                room_type="Lab",
                building="B",
                capacity=30
            ))

        # Playgrounds in Building A
        for i in range(room_config['playground_a']):
            rooms.append(Room(
                room_id=f"A-PG-{i+1:02d}",
                room_type="Playground",
                building="A",
                capacity=40
            ))

        # Playgrounds in Building B
        for i in range(room_config['playground_b']):
            rooms.append(Room(
                room_id=f"B-PG-{i+1:02d}",
                room_type="Playground",
                building="B",
                capacity=40
            ))

        # Music in Building A
        for i in range(room_config['music_a']):
            rooms.append(Room(
                room_id=f"A-MUS-{i+1:02d}",
                room_type="Music",
                building="A",
                capacity=30
            ))

        # Music in Building B
        for i in range(room_config['music_b']):
            rooms.append(Room(
                room_id=f"B-MUS-{i+1:02d}",
                room_type="Music",
                building="B",
                capacity=30
            ))

        # Computers in Building A
        for i in range(room_config['computers_a']):
            rooms.append(Room(
                room_id=f"A-COM-{i+1:02d}",
                room_type="Computer",
                building="A",
                capacity=30
            ))

        # Computers in Building B
        for i in range(room_config['computers_b']):
            rooms.append(Room(
                room_id=f"B-COM-{i+1:02d}",
                room_type="Computer",
                building="B",
                capacity=30
            ))

        # Libraries
        rooms.append(Room(
            room_id="A-LIB-01",
            room_type="Library",
            building="A",
            capacity=30
        ))
        rooms.append(Room(
            room_id="B-LIB-01",
            room_type="Library",
            building="B",
            capacity=30
        ))

        return rooms

    # --------- Curriculum ---------

    def generate_curriculum(self, classes: List[ClassInfo]) -> pd.DataFrame:
        """Generate curriculum requirements for all classes (optimized)."""
        curriculum_data: List[Dict[str, Any]] = []

        # Pre-determine grade keys
        grade_keys: Dict[str, str] = {}
        for class_info in classes:
            if class_info.grade <= 3:
                grade_keys[class_info.class_id] = "g1_3"
            elif class_info.grade <= 6:
                grade_keys[class_info.class_id] = "g4_6"
            else:
                grade_keys[class_info.class_id] = "g7_9"

        # Generate curriculum data
        for class_info in classes:
            grade_key = grade_keys[class_info.class_id]
            for subject, periods in self.config['curriculum']['subjects'].items():
                periods_for_grade = periods.get(grade_key, 0)
                if periods_for_grade and periods_for_grade > 0:
                    curriculum_data.append({
                        'class_id': class_info.class_id,
                        'subject': subject,
                        'periods_per_week': periods_for_grade
                    })

        return pd.DataFrame(curriculum_data)

    # --------- Availability ---------

    def generate_teacher_availability(self, teachers: List[Teacher]) -> pd.DataFrame:
        """Generate teacher availability matrix (optimized for speed)."""
        days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu']
        slots = list(range(1, self.config['schedule']['slots_per_day'] + 1))

        total_combinations = len(teachers) * len(days) * len(slots)
        availability_flags = np.random.random(total_combinations) < 0.85

        teacher_ids: List[str] = []
        day_list: List[str] = []
        slot_list: List[int] = []

        for teacher in teachers:
            for day in days:
                for slot in slots:
                    teacher_ids.append(teacher.teacher_id)
                    day_list.append(day)
                    slot_list.append(slot)

        availability_data = pd.DataFrame({
            'teacher_id': teacher_ids,
            'day': day_list,
            'slot': slot_list,
            'available': availability_flags
        })
        return availability_data

    # --------- Save ---------

    def save_data(self,
                  classes: List[ClassInfo],
                  teachers: List[Teacher],
                  rooms: List[Room],
                  curriculum: pd.DataFrame,
                  teacher_availability: pd.DataFrame):
        """Save all generated data to CSV files."""
        # Convert classes to DataFrame
        classes_df = pd.DataFrame([{
            'class_id': c.class_id,
            'grade': c.grade,
            'section': c.section,
            'gender': c.gender,
            'building': c.building,
            'periods_per_week': c.periods_per_week
        } for c in classes])

        # Convert teachers to DataFrame
        teachers_df = pd.DataFrame([{
            'teacher_id': t.teacher_id,
            'name': t.name,
            'gender': t.gender,
            'primary_subject': t.primary_subject,
            'secondary_subjects': ','.join(t.secondary_subjects),
            'home_building': t.home_building,
            'daily_class_required': t.daily_class_required,
            'min_periods': t.min_periods,
            'max_periods': t.max_periods
        } for t in teachers])

        # Convert rooms to DataFrame
        rooms_df = pd.DataFrame([{
            'room_id': r.room_id,
            'room_type': r.room_type,
            'building': r.building,
            'capacity': r.capacity
        } for r in rooms])

        # Save all DataFrames
        classes_df.to_csv(self.config['paths']['input_files']['classes'], index=False)
        teachers_df.to_csv(self.config['paths']['input_files']['teachers'], index=False)
        rooms_df.to_csv(self.config['paths']['input_files']['rooms'], index=False)
        curriculum.to_csv(self.config['paths']['input_files']['curriculum'], index=False)
        teacher_availability.to_csv(self.config['paths']['input_files']['teacher_availability'], index=False)

        # Print summary with grade-based statistics
        specialized_count = sum(1 for t in teachers if t.primary_subject in self.specialized_subjects)
        
        # Calculate grade-based statistics
        grade_stats = {}
        for class_info in classes:
            grade_cat = 'g7_9' if class_info.grade >= 7 else ('g4_6' if class_info.grade >= 4 else 'g1_3')
            if grade_cat not in grade_stats:
                grade_stats[grade_cat] = {'classes': 0, 'boys': 0, 'girls': 0}
            grade_stats[grade_cat]['classes'] += 1
            if class_info.gender == 'Boys':
                grade_stats[grade_cat]['boys'] += 1
            else:
                grade_stats[grade_cat]['girls'] += 1
        
        print(f" Data generation completed!")
        print(f"   - Classes: {len(classes)}")
        print(f"   - Teachers: {len(teachers)} (including {specialized_count} specialized)")
        print(f"   - Rooms: {len(rooms)}")
        print(f"   - Curriculum entries: {len(curriculum)}")
        print(f"   - Availability entries: {len(teacher_availability)}")
        
        print(f"\nGrade-based Class Distribution:")
        for grade_cat in ['g1_3', 'g4_6', 'g7_9']:
            if grade_cat in grade_stats:
                stats = grade_stats[grade_cat]
                print(f"   {grade_cat.upper()}: {stats['classes']} classes "
                      f"({stats['boys']} boys, {stats['girls']} girls)")
        
        print(f"\nSpecialized Teacher Targets:")
        for sub, cnt in self.specialized_teachers_per_subject.items():
            print(f"       • {sub}: {cnt}")
        
        print(f"\nSubject Combinations Matrix:")
        for combo_name, combo_data in self.subject_combinations.items():
            primary = combo_data.get('primary_subject', 'N/A')
            secondary = combo_data.get('secondary_subject', 'N/A')
            score = combo_data.get('compatibility_score', 0.0)
            print(f"       • {primary} + {secondary}: {score:.1f}")
        
        print(f"   - Data saved to: {self.config['paths']['data_dir']}")

# ----------------------------
# Main
# ----------------------------

def main():
    """Main function for data generation."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'system_config.yaml')

    print(" Starting Step 1: Data Generation")
    print("=" * 50)

    try:
        # Initialize generator
        generator = DataGenerator(config_path)

        # Generate all data
        print(" Generating classes...")
        classes = generator.generate_classes()
        print(f"   Generated {len(classes)} classes")

        print(" Generating teachers...")
        teachers = generator.generate_teachers()
        print(f"   Generated {len(teachers)} teachers")

        print(" Generating rooms...")
        rooms = generator.generate_rooms()
        print(f"   Generated {len(rooms)} rooms")

        print(" Generating curriculum...")
        curriculum = generator.generate_curriculum(classes)
        print(f"   Generated {len(curriculum)} curriculum entries")

        print(" Generating teacher availability...")
        teacher_availability = generator.generate_teacher_availability(teachers)
        print(f"   Generated {len(teacher_availability)} availability entries")

        # Save all data
        print(" Saving data...")
        generator.save_data(classes, teachers, rooms, curriculum, teacher_availability)

        print("\n Step 1 completed successfully!")
        print("   Ready for Step 2: Feature Engineering")

    except Exception as e:
        print(f"\n Error in data generation: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    main()
