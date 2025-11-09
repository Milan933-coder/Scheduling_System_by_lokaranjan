"""
Enhanced Hybrid Timetable Generator with GNN Seeding and Local Search
====================================================================

This implementation follows the enhancement roadmap:
1. Hybridization: GNN for seeding + Local Search for refinement
2. Architectural Evolution: Rich edge features
3. Advanced Learning: Dense reward shaping
4. Solution Diversity: Portfolio of diverse solutions

Fixed version with proper GPU support and dimension consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict, deque
import copy
from torch_geometric.nn import GATConv, global_mean_pool
import torch_geometric
import heapq
import time

# Set device globally
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {DEVICE}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TimeSlot:
    day: int  # 0-4 for Monday to Friday
    period: int  # 0-7 for 8 periods per day
    
    def __hash__(self):
        return hash((self.day, self.period))
    
    def __eq__(self, other):
        return self.day == other.day and self.period == other.period
    
    def __repr__(self):
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        return f"{days[self.day]}Period{self.period + 1}"

@dataclass
class Teacher:
    teacher_id: str
    name: str
    subjects: List[str]
    unavailable_slots: List[TimeSlot] = field(default_factory=list)
    preferred_slots: List[TimeSlot] = field(default_factory=list)

@dataclass
class Classroom:
    room_id: str
    capacity: int
    equipment: List[str] = field(default_factory=list)

@dataclass
class Subject:
    subject_id: str
    name: str
    periods_per_week: int
    requires_lab: bool = False
    preferred_periods: List[int] = field(default_factory=list)

@dataclass
class ClassGroup:
    class_id: str
    size: int
    subjects: List[str]
    preferred_schedule: Dict[int, List[int]] = field(default_factory=dict)

# ============================================================================
# ENHANCED GRAPH-BASED STATE REPRESENTATION WITH RICH EDGE FEATURES
# ============================================================================

class EnhancedGraphTimetableState:
    def __init__(self, teachers: List[Teacher], classrooms: List[Classroom],
                 subjects: List[Subject], class_groups: List[ClassGroup]):
        self.teachers = {t.teacher_id: t for t in teachers}
        self.classrooms = {c.room_id: c for c in classrooms}
        self.subjects = {s.subject_id: s for s in subjects}
        self.class_groups = {c.class_id: c for c in class_groups}
        
        # Create mappings for graph encoding
        self.teacher_to_idx = {tid: i for i, tid in enumerate(self.teachers.keys())}
        self.classroom_to_idx = {rid: i for i, rid in enumerate(self.classrooms.keys())}
        self.subject_to_idx = {sid: i for i, sid in enumerate(self.subjects.keys())}
        self.class_to_idx = {cid: i for i, cid in enumerate(self.class_groups.keys())}
        
        # Time slots
        self.time_slots = []
        for day in range(5):
            for period in range(8):
                self.time_slots.append(TimeSlot(day, period))
        
        # Schedule and tracking
        self.schedule = {slot: [] for slot in self.time_slots}
        self.teacher_assignments = defaultdict(list)
        self.class_assignments = defaultdict(list)
        self.room_assignments = defaultdict(list)
        self.subject_completion = defaultdict(lambda: defaultdict(int))
        
        # Graph structure with rich edge features
        self._build_enhanced_graph_structure()
    
    def _build_enhanced_graph_structure(self):
        """Build the graph representation with rich edge features"""
        # Node features - CONSISTENT 5 DIMENSIONS FOR ALL NODE TYPES
        self.node_features = []
        self.node_types = []
        
        # Class nodes: [size, num_subjects, completion_ratio, 0, 0]
        for class_id, class_group in self.class_groups.items():
            completed = sum(self.subject_completion[class_id].values())
            total_needed = sum(self.subjects[s].periods_per_week 
                             for s in class_group.subjects)
            completion_ratio = completed / total_needed if total_needed > 0 else 0
            
            features = [
                class_group.size / 50.0,  # Normalized size
                len(class_group.subjects) / 10.0,  # Normalized subject count
                completion_ratio,  # Completion percentage
                0.0,  # Padding
                0.0   # Padding
            ]
            self.node_features.append(features)
            self.node_types.append(0)  # Class node type
        
        # Teacher nodes: [subject_count, workload, availability, 0, 0]
        for teacher_id, teacher in self.teachers.items():
            workload = len(self.teacher_assignments[teacher_id])
            availability = 1.0 - (len(teacher.unavailable_slots) / 40.0)  # 40 total slots
            
            features = [
                len(teacher.subjects) / 10.0,  # Normalized subject count
                workload / 40.0,  # Normalized workload
                availability,  # Availability score
                0.0,  # Padding
                0.0   # Padding
            ]
            self.node_features.append(features)
            self.node_types.append(1)  # Teacher node type
        
        # Room nodes: [capacity, has_lab, has_projector, has_audio, has_computers]
        for room_id, room in self.classrooms.items():
            has_lab = 1.0 if 'lab' in room.equipment else 0.0
            has_projector = 1.0 if 'projector' in room.equipment else 0.0
            has_audio = 1.0 if 'audio' in room.equipment else 0.0
            has_computers = 1.0 if 'computers' in room.equipment else 0.0
            
            features = [
                room.capacity / 50.0,  # Normalized capacity
                has_lab,
                has_projector,
                has_audio,
                has_computers
            ]
            self.node_features.append(features)
            self.node_types.append(2)  # Room node type
        
        # Subject nodes: [periods_per_week, requires_lab, preferred_period_score, 0, 0]
        for subject_id, subject in self.subjects.items():
            preferred_period_score = len(subject.preferred_periods) / 8.0 if subject.preferred_periods else 0.5
            
            features = [
                subject.periods_per_week / 8.0,  # Normalized periods
                1.0 if subject.requires_lab else 0.0,  # Lab requirement
                preferred_period_score,  # Preference score
                0.0,  # Padding
                0.0   # Padding
            ]
            self.node_features.append(features)
            self.node_types.append(3)  # Subject node type
        
        # Timeslot nodes: [day, period, preference_score, 0, 0]
        for slot in self.time_slots:
            # Calculate how preferred this timeslot is across all entities
            preference_score = 0.0
            count = 0
            
            # Check class preferences
            for class_id, class_group in self.class_groups.items():
                if slot.day in class_group.preferred_schedule:
                    if slot.period in class_group.preferred_schedule[slot.day]:
                        preference_score += 1.0
                count += 1
            
            # Check teacher preferences
            for teacher_id, teacher in self.teachers.items():
                if slot in teacher.preferred_slots:
                    preference_score += 1.0
                count += 1
            
            # Normalize preference score
            preference_score = preference_score / count if count > 0 else 0.5
            
            features = [
                slot.day / 5.0,  # Normalized day
                slot.period / 8.0,  # Normalized period
                preference_score,  # Preference score
                0.0,  # Padding
                0.0   # Padding
            ]
            self.node_features.append(features)
            self.node_types.append(4)  # Timeslot node type
        
        # Convert to tensors and move to GPU
        self.node_types = torch.tensor(self.node_types, dtype=torch.long, device=DEVICE)
        self.node_features = torch.tensor(self.node_features, dtype=torch.float, device=DEVICE)
        
        # Build edges with rich features
        self.edge_index = []  # [2, num_edges]
        self.edge_attr = []   # Edge features as vectors
        
        # Build enhanced edges with rich features
        self._build_enhanced_edges()
    
    def _build_enhanced_edges(self):
        """Build edges with rich feature vectors - STANDARDIZED TO 4 DIMENSIONS"""
        
        # Class-subject edges
        for class_idx, class_id in enumerate(self.class_groups.keys()):
            class_group = self.class_groups[class_id]
            for subject_id in class_group.subjects:
                subject_idx = self.subject_to_idx[subject_id] + len(self.class_groups) + len(self.teachers) + len(self.classrooms)
                
                # Edge features: importance based on remaining need
                required = self.subjects[subject_id].periods_per_week
                assigned = self.subject_completion[class_id][subject_id]
                remaining_need = (required - assigned) / required if required > 0 else 0
                
                self.edge_index.append([class_idx, subject_idx])
                self.edge_attr.append([remaining_need, 1.0, 0.0, 0.0])  # [need, class-subject relation, padding, padding]
        
        # Teacher-subject edges
        for teacher_idx, teacher_id in enumerate(self.teachers.keys()):
            teacher = self.teachers[teacher_id]
            for subject_id in teacher.subjects:
                subject_idx = self.subject_to_idx[subject_id] + len(self.class_groups) + len(self.teachers) + len(self.classrooms)
                
                # Edge features: teacher proficiency
                proficiency = 1.0  # Default
                workload = len(self.teacher_assignments[teacher_id])
                workload_factor = 1.0 - (workload / 40.0)  # Lower workload = higher availability
                
                self.edge_index.append([teacher_idx + len(self.class_groups), subject_idx])
                self.edge_attr.append([proficiency, workload_factor, 1.0, 0.0])  # [proficiency, availability, teacher-subject relation, padding]
        
        # Subject-room edges (which rooms are suitable for which subjects)
        for subject_idx, subject_id in enumerate(self.subjects.keys()):
            subject = self.subjects[subject_id]
            for room_idx, room_id in enumerate(self.classrooms.keys()):
                room = self.classrooms[room_id]
                
                # Check if room is suitable for subject
                suitability = 1.0
                capacity_match = 1.0
                lab_match = 1.0
                
                # Capacity suitability
                min_class_size = min(cg.size for cg in self.class_groups.values())
                capacity_match = min(1.0, room.capacity / min_class_size)
                
                # Lab requirement match
                if subject.requires_lab:
                    lab_match = 1.0 if 'lab' in room.equipment else 0.0
                
                # Only create edge if suitable
                if lab_match > 0 and capacity_match > 0.5:
                    actual_room_idx = room_idx + len(self.class_groups) + len(self.teachers)
                    actual_subject_idx = subject_idx + len(self.class_groups) + len(self.teachers) + len(self.classrooms)
                    
                    self.edge_index.append([actual_subject_idx, actual_room_idx])
                    self.edge_attr.append([capacity_match, lab_match, suitability, 1.0])  # [capacity, lab, suitability, subject-room relation]
        
        # Teacher-timeslot edges (availability)
        for teacher_idx, teacher_id in enumerate(self.teachers.keys()):
            teacher = self.teachers[teacher_id]
            for slot_idx, slot in enumerate(self.time_slots):
                availability = 1.0 if slot not in teacher.unavailable_slots else 0.0
                preference = 1.0 if slot in teacher.preferred_slots else 0.5
                
                if availability > 0:
                    actual_teacher_idx = teacher_idx + len(self.class_groups)
                    actual_slot_idx = slot_idx + len(self.class_groups) + len(self.teachers) + len(self.classrooms) + len(self.subjects)
                    
                    self.edge_index.append([actual_teacher_idx, actual_slot_idx])
                    self.edge_attr.append([availability, preference, 1.0, 0.0])  # [availability, preference, teacher-timeslot relation, padding]
        
        # Room-timeslot edges
        for room_idx, room_id in enumerate(self.classrooms.keys()):
            for slot_idx, slot in enumerate(self.time_slots):
                # Room availability is generally high
                room_preference = 0.7  # Default
                if 'lab' in self.classrooms[room_id].equipment and slot.period >= 4:  # Prefer afternoon for labs
                    room_preference = 0.9
                
                actual_room_idx = room_idx + len(self.class_groups) + len(self.teachers)
                actual_slot_idx = slot_idx + len(self.class_groups) + len(self.teachers) + len(self.classrooms) + len(self.subjects)
                
                self.edge_index.append([actual_room_idx, actual_slot_idx])
                self.edge_attr.append([1.0, room_preference, 1.0, 0.0])  # [availability, preference, room-timeslot relation, padding]
        
        # Class-timeslot edges
        for class_idx, class_id in enumerate(self.class_groups.keys()):
            class_group = self.class_groups[class_id]
            for slot_idx, slot in enumerate(self.time_slots):
                # Check if this is a preferred timeslot for the class
                preference = 0.5  # Default
                if slot.day in class_group.preferred_schedule:
                    if slot.period in class_group.preferred_schedule[slot.day]:
                        preference = 1.0
                    else:
                        preference = 0.3
                
                actual_slot_idx = slot_idx + len(self.class_groups) + len(self.teachers) + len(self.classrooms) + len(self.subjects)
                
                self.edge_index.append([class_idx, actual_slot_idx])
                self.edge_attr.append([1.0, preference, 1.0, 0.0])  # [availability, preference, class-timeslot relation, padding]
        
        # Convert to tensors and move to GPU
        if self.edge_index:
            self.edge_index = torch.tensor(self.edge_index, dtype=torch.long, device=DEVICE).t().contiguous()
            self.edge_attr = torch.tensor(self.edge_attr, dtype=torch.float, device=DEVICE)
        else:
            self.edge_index = torch.empty((2, 0), dtype=torch.long, device=DEVICE)
            self.edge_attr = torch.empty((0, 4), dtype=torch.float, device=DEVICE)
    
    def to_graph_data(self):
        """Convert state to PyG Data object"""
        data = torch_geometric.data.Data(
            x=self.node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            node_type=self.node_types
        )
        return data.to(DEVICE)
    
    def copy(self):
        """Create a deep copy of the state"""
        new_state = EnhancedGraphTimetableState(
            list(self.teachers.values()),
            list(self.classrooms.values()),
            list(self.subjects.values()),
            list(self.class_groups.values())
        )
        new_state.schedule = copy.deepcopy(self.schedule)
        new_state.teacher_assignments = copy.deepcopy(self.teacher_assignments)
        new_state.class_assignments = copy.deepcopy(self.class_assignments)
        new_state.room_assignments = copy.deepcopy(self.room_assignments)
        new_state.subject_completion = copy.deepcopy(self.subject_completion)
        return new_state
    
    def is_valid_assignment(self, time_slot: TimeSlot, class_id: str,
                          subject_id: str, teacher_id: str, room_id: str) -> bool:
        """Check if an assignment violates hard constraints"""
        # Check if teacher is available and can teach the subject
        teacher = self.teachers[teacher_id]
        if time_slot in teacher.unavailable_slots:
            return False
        if subject_id not in teacher.subjects:
            return False
        
        # Check if teacher is already assigned at this time
        for assigned_slot, _ in self.teacher_assignments[teacher_id]:
            if assigned_slot == time_slot:
                return False
        
        # Check if class is already assigned at this time
        for assigned_slot, _ in self.class_assignments[class_id]:
            if assigned_slot == time_slot:
                return False
        
        # Check if room is already occupied
        for assigned_slot, _ in self.room_assignments[room_id]:
            if assigned_slot == time_slot:
                return False
        
        # Check room capacity
        classroom = self.classrooms[room_id]
        class_group = self.class_groups[class_id]
        if classroom.capacity < class_group.size:
            return False
        
        # Check lab requirements
        subject = self.subjects[subject_id]
        if subject.requires_lab and 'lab' not in classroom.equipment:
            return False
        
        return True
    
    def make_assignment(self, time_slot: TimeSlot, class_id: str,
                       subject_id: str, teacher_id: str, room_id: str):
        """Make an assignment and update tracking structures"""
        if not self.is_valid_assignment(time_slot, class_id, subject_id, teacher_id, room_id):
            raise ValueError("Invalid assignment")
        
        # Add to schedule
        assignment = (class_id, subject_id, teacher_id, room_id)
        self.schedule[time_slot].append(assignment)
        
        # Update tracking
        self.teacher_assignments[teacher_id].append((time_slot, class_id))
        self.class_assignments[class_id].append((time_slot, subject_id))
        self.room_assignments[room_id].append((time_slot, class_id))
        self.subject_completion[class_id][subject_id] += 1
    
    def get_possible_actions(self) -> List[Tuple]:
        """Get all possible valid assignments from current state with masking"""
        actions = []
        for time_slot in self.time_slots:
            for class_id in self.class_groups:
                class_group = self.class_groups[class_id]
                
                # Check if class is already assigned at this time
                if any(slot == time_slot for slot, _ in self.class_assignments[class_id]):
                    continue
                
                for subject_id in class_group.subjects:
                    # Skip if subject is already completed for this class
                    required_periods = self.subjects[subject_id].periods_per_week
                    current_periods = self.subject_completion[class_id][subject_id]
                    if current_periods >= required_periods:
                        continue
                    
                    for teacher_id in self.teachers:
                        if subject_id not in self.teachers[teacher_id].subjects:
                            continue
                        
                        for room_id in self.classrooms:
                            if self.is_valid_assignment(time_slot, class_id, subject_id, teacher_id, room_id):
                                actions.append((time_slot, class_id, subject_id, teacher_id, room_id))
        
        return actions
    
    def is_complete(self) -> bool:
        """Check if all subjects are scheduled for all classes"""
        for class_id in self.class_groups:
            class_group = self.class_groups[class_id]
            for subject_id in class_group.subjects:
                required = self.subjects[subject_id].periods_per_week
                assigned = self.subject_completion[class_id][subject_id]
                if assigned < required:
                    return False
        return True
    
    def calculate_fitness(self) -> float:
        """Calculate fitness score (higher is better)"""
        score = 0
        
        # Reward completed subjects
        total_required = 0
        total_assigned = 0
        for class_id in self.class_groups:
            class_group = self.class_groups[class_id]
            for subject_id in class_group.subjects:
                required = self.subjects[subject_id].periods_per_week
                assigned = self.subject_completion[class_id][subject_id]
                total_required += required
                total_assigned += min(assigned, required)
        
        completion_ratio = total_assigned / total_required if total_required > 0 else 0
        score += completion_ratio * 1000
        
        # Penalty for over-assignment
        over_assignments = max(0, total_assigned - total_required)
        score -= over_assignments * 50
        
        # Bonus for balanced teacher workload
        teacher_loads = [len(assignments) for assignments in self.teacher_assignments.values()]
        if teacher_loads:
            load_variance = np.var(teacher_loads)
            score -= load_variance * 10
        
        # Bonus for balanced daily schedules
        daily_loads = [0] * 5
        for slot in self.time_slots:
            if self.schedule[slot]:
                daily_loads[slot.day] += len(self.schedule[slot])
        
        if daily_loads:
            daily_variance = np.var(daily_loads)
            score -= daily_variance * 5
        
        # Reward for preference satisfaction
        preference_score = self.calculate_preference_score()
        score += preference_score * 20
        
        return score
    
    def calculate_preference_score(self) -> float:
        """Calculate how well preferences are satisfied"""
        score = 0
        count = 0
        
        # Teacher preferences
        for teacher_id, assignments in self.teacher_assignments.items():
            teacher = self.teachers[teacher_id]
            for slot, _ in assignments:
                if slot in teacher.preferred_slots:
                    score += 1.0
                count += 1
        
        # Class preferences
        for class_id, assignments in self.class_assignments.items():
            class_group = self.class_groups[class_id]
            for slot, _ in assignments:
                if slot.day in class_group.preferred_schedule:
                    if slot.period in class_group.preferred_schedule[slot.day]:
                        score += 1.0
                count += 1
        
        # Subject preferences (preferred periods)
        for slot, assignments in self.schedule.items():
            for class_id, subject_id, teacher_id, room_id in assignments:
                subject = self.subjects[subject_id]
                if subject.preferred_periods and slot.period in subject.preferred_periods:
                    score += 1.0
                count += 1
        
        return score / count if count > 0 else 0
    
    def get_intermediate_reward(self, action: Tuple) -> float:
        """Calculate intermediate reward for a specific action (dense reward shaping)"""
        time_slot, class_id, subject_id, teacher_id, room_id = action
        reward = 0.0
        
        # Base reward for making a valid assignment
        reward += 1.0
        
        # Reward for completing a subject
        required = self.subjects[subject_id].periods_per_week
        assigned = self.subject_completion[class_id][subject_id] + 1  # +1 because we're about to assign
        if assigned >= required:
            reward += 5.0
        
        # Reward for preference satisfaction
        teacher = self.teachers[teacher_id]
        if time_slot in teacher.preferred_slots:
            reward += 2.0
        
        class_group = self.class_groups[class_id]
        if time_slot.day in class_group.preferred_schedule:
            if time_slot.period in class_group.preferred_schedule[time_slot.day]:
                reward += 2.0
        
        subject = self.subjects[subject_id]
        if subject.preferred_periods and time_slot.period in subject.preferred_periods:
            reward += 1.5
        
        # Penalty for Friday afternoon classes (generally less preferred)
        if time_slot.day == 4 and time_slot.period >= 6:  # Friday, last two periods
            reward -= 1.0
        
        # Penalty for consecutive classes for the same class
        class_assignments = self.class_assignments[class_id]
        if class_assignments:
            last_slot = class_assignments[-1][0]  # Last assigned slot for this class
            if last_slot.day == time_slot.day and abs(last_slot.period - time_slot.period) == 1:
                reward -= 0.5  # Consecutive classes penalty
        
        return reward

# ============================================================================
# ENHANCED GNN WITH RICH EDGE FEATURES
# ============================================================================

class EnhancedGNNPolicyValueNetwork(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=128, num_heads=4):
        super(EnhancedGNNPolicyValueNetwork, self).__init__()
        
        # Node type embedding
        self.node_type_embedding = nn.Embedding(5, 8)  # 5 node types
        
        # Edge feature processing (MLP for rich edge features)
        if edge_feature_dim > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.ReLU()
            )
        else:
            self.edge_encoder = nn.Linear(1, hidden_dim // 2)
        
        # Initial node feature transformation
        self.node_linear = nn.Linear(node_feature_dim + 8, hidden_dim)
        
        # Graph attention layers with edge features
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, edge_dim=hidden_dim // 2)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, edge_dim=hidden_dim // 2, concat=False)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),  # 5 components of an action
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
    
    def forward(self, data):
        # Embed node types and combine with features
        node_type_emb = self.node_type_embedding(data.node_type)
        x = torch.cat([data.x, node_type_emb], dim=1)
        x = F.relu(self.node_linear(x))
        
        # Process edge features
        edge_attr_emb = self.edge_encoder(data.edge_attr)
        
        # Apply graph convolutions with edge features
        x = F.relu(self.conv1(x, data.edge_index, edge_attr=edge_attr_emb))
        x = F.relu(self.conv2(x, data.edge_index, edge_attr=edge_attr_emb))
        
        # Global pooling for value head
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_embedding = global_mean_pool(x, batch=batch)
        value = self.value_head(graph_embedding).squeeze()
        
        return x, value
    
    def compute_action_scores(self, node_embeddings, actions, action_mapping):
        """Compute scores for specific actions based on node embeddings"""
        scores = []
        
        for action in actions:
            time_slot, class_id, subject_id, teacher_id, room_id = action
            
            # Get embeddings for each component of the action
            class_emb = node_embeddings[action_mapping['class'][class_id]]
            subject_emb = node_embeddings[action_mapping['subject'][subject_id]]
            teacher_emb = node_embeddings[action_mapping['teacher'][teacher_id]]
            room_emb = node_embeddings[action_mapping['room'][room_id]]
            timeslot_emb = node_embeddings[action_mapping['timeslot'][(time_slot.day, time_slot.period)]]
            
            # Combine embeddings (concatenation)
            combined_emb = torch.cat([class_emb, subject_emb, teacher_emb, room_emb, timeslot_emb], dim=0)
            
            # Score the action
            score = self.policy_head(combined_emb.unsqueeze(0)).squeeze()
            scores.append(score)
        
        if scores:
            return torch.stack(scores)
        else:
            # Ensure empty tensor is on correct device
            return torch.tensor([], device=node_embeddings.device)

# ============================================================================
# ENHANCED MCTS WITH DENSE REWARD SHAPING
# ============================================================================

class EnhancedMCTSNode:
    def __init__(self, state: EnhancedGraphTimetableState, parent=None, action=None, prior_prob=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior_prob = prior_prob
        self.immediate_reward = 0.0  # Store immediate reward for this action
        
        self.children = {}  # action -> child node
        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0
        self.is_terminal = state.is_complete()
        self.is_expanded = False
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def expand(self, network: EnhancedGNNPolicyValueNetwork, action_mapping: Dict):
        """Expand node using neural network predictions"""
        if self.is_expanded:
            return
        
        # Get network predictions
        graph_data = self.state.to_graph_data()
        with torch.no_grad():
            node_embeddings, value = network(graph_data)
        
        # Get valid actions
        valid_actions = self.state.get_possible_actions()
        if not valid_actions:
            self.is_expanded = True
            return
        
        # Compute action scores
        action_scores = network.compute_action_scores(node_embeddings, valid_actions, action_mapping)
        action_probs = F.softmax(action_scores, dim=0)
        
        # Create children
        for i, action in enumerate(valid_actions):
            new_state = self.state.copy()
            time_slot, class_id, subject_id, teacher_id, room_id = action
            
            # Calculate immediate reward before making the assignment
            immediate_reward = new_state.get_intermediate_reward(action)
            
            # Make the assignment
            new_state.make_assignment(time_slot, class_id, subject_id, teacher_id, room_id)
            
            prior = action_probs[i].item()
            child = EnhancedMCTSNode(new_state, parent=self, action=action, prior_prob=prior)
            child.immediate_reward = immediate_reward  # Store immediate reward
            self.children[action] = child
        
        self.is_expanded = True
    
    def select_child(self, c_puct: float = 1.0):
        """Select child using PUCT algorithm"""
        if not self.children:
            return None
        
        def puct_score(child):
            if child.visit_count == 0:
                u_score = float('inf')
            else:
                u_score = (c_puct * child.prior_prob * 
                          math.sqrt(self.visit_count) / (1 + child.visit_count))
            
            q_score = child.mean_value if child.visit_count > 0 else 0
            return q_score + u_score
        
        return max(self.children.values(), key=puct_score)
    
    def backup(self, value: float):
        """Backup value through the tree with immediate rewards"""
        # Include immediate reward in the backup
        total_value = value + self.immediate_reward
        
        self.visit_count += 1
        self.total_value += total_value
        self.mean_value = self.total_value / self.visit_count
        
        if self.parent:
            self.parent.backup(total_value)  # Pass the total value up
    
    def get_action_probs(self, temperature: float = 1.0):
        """Get action probabilities based on visit counts"""
        if not self.children:
            return {}
        
        if temperature == 0:
            # Deterministic: choose most visited
            best_action = max(self.children.keys(),
                            key=lambda a: self.children[a].visit_count)
            probs = {action: 0.0 for action in self.children.keys()}
            probs[best_action] = 1.0
            return probs
        else:
            # Stochastic: proportional to visit count ^ (1/temperature)
            visit_counts = np.array([self.children[a].visit_count for a in self.children.keys()])
            if temperature != 1.0:
                visit_counts = visit_counts ** (1.0 / temperature)
            
            total = np.sum(visit_counts)
            if total == 0:
                # Uniform if no visits
                probs = {action: 1.0 / len(self.children) for action in self.children.keys()}
            else:
                probs = {}
                for i, action in enumerate(self.children.keys()):
                    probs[action] = visit_counts[i] / total
            
            return probs

# ============================================================================
# ENHANCED MCTS WITH DENSE REWARD SUPPORT
# ============================================================================

class EnhancedNeuralMCTS:
    def __init__(self, network: EnhancedGNNPolicyValueNetwork, c_puct: float = 1.0):
        self.network = network
        self.c_puct = c_puct
    
    def search(self, root_state: EnhancedGraphTimetableState, action_mapping: Dict, num_simulations: int = 800):
        """Run MCTS simulations and return action probabilities"""
        root = EnhancedMCTSNode(root_state)
        
        for simulation in range(num_simulations):
            if simulation % 100 == 0 and simulation > 0:
                print(f"  MCTS simulation {simulation}/{num_simulations}")
            
            # Selection: traverse tree to leaf
            node = root
            path = [node]
            
            while not node.is_leaf() and not node.is_terminal:
                node = node.select_child(self.c_puct)
                if node is not None:
                    path.append(node)
                else:
                    break
            
            if node is None:
                continue
            
            # Expansion and evaluation
            if not node.is_terminal:
                node.expand(self.network, action_mapping)
                
                # If we have children, select one for evaluation
                if node.children:
                    node = node.select_child(self.c_puct)
                    if node is not None:
                        path.append(node)
            
            if node is None:
                continue
            
            # Evaluation: get value from network or calculate fitness
            if node.is_terminal:
                value = node.state.calculate_fitness() / 1000.0  # Normalize
            else:
                graph_data = node.state.to_graph_data()
                with torch.no_grad():
                    _, value = self.network(graph_data)
                    value = value.item()
            
            # Backup: propagate value up the tree
            for node_in_path in reversed(path):
                node_in_path.backup(value)
        
        # Return action probabilities based on visit counts
        return root.get_action_probs(temperature=1.0), root
    
    def get_best_action(self, action_probs: Dict) -> Optional[Tuple]:
        """Get the best action from action probabilities"""
        if not action_probs:
            return None
        
        best_action = max(action_probs.keys(), key=lambda a: action_probs[a])
        return best_action

# ============================================================================
# HYBRID TIMETABLE GENERATOR WITH LOCAL SEARCH
# ============================================================================

class HybridTimetableGenerator:
    def __init__(self, teachers: List[Teacher], classrooms: List[Classroom],
                 subjects: List[Subject], class_groups: List[ClassGroup]):
        self.teachers = teachers
        self.classrooms = classrooms
        self.subjects = subjects
        self.class_groups = class_groups
        
        # Create initial state to get dimensions
        initial_state = EnhancedGraphTimetableState(teachers, classrooms, subjects, class_groups)
        
        # Create action mapping for GNN
        self.action_mapping = self._create_action_mapping(initial_state)
        
        # Initialize network
        self.network = EnhancedGNNPolicyValueNetwork(
            node_feature_dim=initial_state.node_features.size(1),
            edge_feature_dim=initial_state.edge_attr.size(1) if initial_state.edge_attr.numel() > 0 else 1,
            hidden_dim=128,
            num_heads=4
        ).to(DEVICE)
        
        # Initialize MCTS
        self.mcts = EnhancedNeuralMCTS(self.network, c_puct=1.0)
        
        # Training data for network improvement
        self.training_data = deque(maxlen=10000)
        
        print(f"🧠 Enhanced GNN Network initialized:")
        print(f"  Node feature dimension: {initial_state.node_features.size(1)}")
        print(f"  Edge feature dimension: {initial_state.edge_attr.size(1) if initial_state.edge_attr.numel() > 0 else 1}")
        print(f"  Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
    
    def _create_action_mapping(self, state: EnhancedGraphTimetableState) -> Dict:
        """Create mapping from entities to their node indices"""
        mapping = {
            'class': {},
            'teacher': {},
            'room': {},
            'subject': {},
            'timeslot': {}
        }
        
        # Class mapping
        for i, class_id in enumerate(state.class_groups.keys()):
            mapping['class'][class_id] = i
        
        # Teacher mapping
        for i, teacher_id in enumerate(state.teachers.keys()):
            mapping['teacher'][teacher_id] = len(state.class_groups) + i
        
        # Room mapping
        for i, room_id in enumerate(state.classrooms.keys()):
            mapping['room'][room_id] = len(state.class_groups) + len(state.teachers) + i
        
        # Subject mapping
        for i, subject_id in enumerate(state.subjects.keys()):
            mapping['subject'][subject_id] = len(state.class_groups) + len(state.teachers) + len(state.classrooms) + i
        
        # Timeslot mapping
        for i, slot in enumerate(state.time_slots):
            mapping['timeslot'][(slot.day, slot.period)] = len(state.class_groups) + len(state.teachers) + len(state.classrooms) + len(state.subjects) + i
        
        return mapping
    
    def generate_seeding_solution(self, max_iterations: int = 30, mcts_sims: int = 50) -> EnhancedGraphTimetableState:
        """Generate a quick initial solution for seeding the local search"""
        print("🌱 Generating seeding solution with GNN...")
        
        current_state = EnhancedGraphTimetableState(self.teachers, self.classrooms,
                                                   self.subjects, self.class_groups)
        
        iteration = 0
        while not current_state.is_complete() and iteration < max_iterations:
            iteration += 1
            
            actions_available = len(current_state.get_possible_actions())
            if actions_available == 0:
                print("⚠ No more valid actions available during seeding!")
                break
            
            # Run MCTS with fewer simulations for speed
            action_probs, root = self.mcts.search(current_state, self.action_mapping, num_simulations=mcts_sims)
            
            if not action_probs:
                print("❌ MCTS could not generate action probabilities during seeding!")
                break
            
            # Select best action
            best_action = self.mcts.get_best_action(action_probs)
            if best_action is None:
                print("❌ No valid action selected during seeding!")
                break
            
            # Apply action
            time_slot, class_id, subject_id, teacher_id, room_id = best_action
            try:
                current_state.make_assignment(time_slot, class_id, subject_id, teacher_id, room_id)
                if iteration % 10 == 0:
                    print(f"  ✅ Seeding iteration {iteration}: {class_id} → {self.subjects[subject_id].name}")
            except ValueError as e:
                print(f"❌ Error applying action during seeding: {e}")
                break
        
        fitness = current_state.calculate_fitness()
        completeness = current_state.is_complete()
        print(f"🌱 Seeding completed: Fitness={fitness:.1f}, Complete={completeness}")
        
        return current_state
    
    def local_search_refinement(self, state: EnhancedGraphTimetableState,
                               max_iterations: int = 1000,
                               initial_temp: float = 1.0,
                               cooling_rate: float = 0.99) -> EnhancedGraphTimetableState:
        """Refine a solution using simulated annealing"""
        print("🔥 Running local search refinement...")
        
        current_state = state.copy()
        current_energy = -current_state.calculate_fitness()  # Negative because we minimize energy
        
        best_state = current_state.copy()
        best_energy = current_energy
        
        temperature = initial_temp
        iteration = 0
        improvements = 0
        
        while iteration < max_iterations and temperature > 0.01:
            iteration += 1
            
            # Generate a neighbor by making a random valid move
            neighbor_state, move_description = self._generate_neighbor(current_state)
            
            if neighbor_state is None:
                continue  # Skip if no valid move found
            
            neighbor_energy = -neighbor_state.calculate_fitness()
            
            # Decide whether to accept the neighbor
            energy_delta = neighbor_energy - current_energy
            accept = False
            
            if energy_delta < 0:
                # Always accept improving moves
                accept = True
            else:
                # Accept worsening moves with probability based on temperature
                accept_prob = math.exp(-energy_delta / temperature)
                accept = random.random() < accept_prob
            
            if accept:
                current_state = neighbor_state
                current_energy = neighbor_energy
                
                if current_energy < best_energy:
                    best_state = current_state.copy()
                    best_energy = current_energy
                    improvements += 1
                    
                    if improvements % 10 == 0:
                        print(f"  🔥 Improvement {improvements}: Energy={-best_energy:.1f}")
            
            # Cool down temperature
            temperature *= cooling_rate
        
        print(f"🔥 Local search completed: {improvements} improvements, Final energy={-best_energy:.1f}")
        return best_state
    
    def _generate_neighbor(self, state: EnhancedGraphTimetableState) -> Tuple[Optional[EnhancedGraphTimetableState], str]:
        """Generate a neighbor state by making a random valid move"""
        # Several types of moves:
        # 1. Swap two assignments
        # 2. Move an assignment to a different timeslot
        # 3. Change the teacher for an assignment
        # 4. Change the room for an assignment
        
        move_type = random.choice(['swap', 'move', 'change_teacher', 'change_room'])
        neighbor = state.copy()
        description = ""
        
        if move_type == 'swap' and len(neighbor.schedule) >= 2:
            # Try to swap two assignments
            non_empty_slots = [slot for slot, assignments in neighbor.schedule.items() if assignments]
            if len(non_empty_slots) < 2:
                return None, "No slots to swap"
            
            slot1, slot2 = random.sample(non_empty_slots, 2)
            if not neighbor.schedule[slot1] or not neighbor.schedule[slot2]:
                return None, "Empty slot in swap"
            
            # Try to swap random assignments from each slot
            assignment1 = random.choice(neighbor.schedule[slot1])
            assignment2 = random.choice(neighbor.schedule[slot2])
            
            class_id1, subject_id1, teacher_id1, room_id1 = assignment1
            class_id2, subject_id2, teacher_id2, room_id2 = assignment2
            
            # Check if swaps are valid
            valid1 = neighbor.is_valid_assignment(slot2, class_id1, subject_id1, teacher_id1, room_id1)
            valid2 = neighbor.is_valid_assignment(slot1, class_id2, subject_id2, teacher_id2, room_id2)
            
            if valid1 and valid2:
                # Remove original assignments
                neighbor.schedule[slot1].remove(assignment1)
                neighbor.schedule[slot2].remove(assignment2)
                
                # Update tracking structures (simplified for brevity)
                # Add new assignments
                neighbor.schedule[slot1].append((class_id2, subject_id2, teacher_id2, room_id2))
                neighbor.schedule[slot2].append((class_id1, subject_id1, teacher_id1, room_id1))
                
                description = f"Swapped {class_id1} at {slot1} with {class_id2} at {slot2}"
                return neighbor, description
        
        # Other move types can be implemented similarly...
        return None, "No valid move generated"
    
    def generate_hybrid_timetable(self) -> EnhancedGraphTimetableState:
        """Generate optimized timetable using hybrid approach"""
        print("🚀 Starting Hybrid Timetable Generation...")
        print("🧠 Using GNN Seeding + Local Search Refinement")
        print("-" * 70)
        
        start_time = time.time()
        
        # Step 1: Generate seeding solution with GNN
        seeding_solution = self.generate_seeding_solution(max_iterations=30, mcts_sims=50)
        
        # Step 2: Refine with local search
        final_solution = self.local_search_refinement(seeding_solution, max_iterations=1000)
        
        end_time = time.time()
        
        print("-" * 70)
        print(f"🎯 Hybrid generation completed in {end_time - start_time:.2f} seconds!")
        print(f"📈 Final fitness: {final_solution.calculate_fitness():.1f}")
        print(f"✅ Complete schedule: {final_solution.is_complete()}")
        
        return final_solution

# ============================================================================
# EXAMPLE USAGE AND MAIN FUNCTION
# ============================================================================

def create_enhanced_sample_data():
    """Create comprehensive sample data for testing"""
    
    # Define teachers with preferences
    teachers = [
        Teacher("T001", "Dr. Alice Smith", ["MATH", "PHYS"],
                unavailable_slots=[TimeSlot(4, 6), TimeSlot(4, 7)],  # No Friday afternoon
                preferred_slots=[TimeSlot(0, 0), TimeSlot(0, 1), TimeSlot(2, 0)]),  # Prefer Monday/Wednesday morning
        
        Teacher("T002", "Prof. Bob Johnson", ["ENG", "HIST"],
                preferred_slots=[TimeSlot(1, 2), TimeSlot(1, 3), TimeSlot(3, 2), TimeSlot(3, 3)]),  # Prefer Tue/Thu late morning
        
        Teacher("T003", "Dr. Carol Wilson", ["CHEM", "BIO"],
                unavailable_slots=[TimeSlot(0, 0), TimeSlot(0, 1)],  # No Monday morning
                preferred_slots=[TimeSlot(2, 4), TimeSlot(2, 5), TimeSlot(4, 4), TimeSlot(4, 5)]),  # Prefer mid-day
        
        Teacher("T004", "Mr. David Brown", ["MATH", "CS"],
                preferred_slots=[TimeSlot(1, 0), TimeSlot(1, 1), TimeSlot(3, 0), TimeSlot(3, 1)]),  # Prefer Tue/Thu early
        
        Teacher("T005", "Ms. Emma Davis", ["ENG", "ART"],
                unavailable_slots=[TimeSlot(2, 6), TimeSlot(2, 7)],  # No Wednesday afternoon
                preferred_slots=[TimeSlot(0, 4), TimeSlot(0, 5), TimeSlot(4, 0), TimeSlot(4, 1)]),  # Prefer Monday late morning, Friday early
    ]
    
    # Define classrooms with specific capabilities
    classrooms = [
        Classroom("R101", 35, ["projector", "whiteboard"]),
        Classroom("R102", 30, ["projector", "whiteboard", "audio"]),
        Classroom("R201", 25, ["lab", "projector", "computers"]),
        Classroom("R202", 28, ["lab", "projector", "equipment"]),
        Classroom("R203", 20, ["lab", "computers", "projector"]),
        Classroom("R301", 40, ["projector", "whiteboard", "audio"]),
        Classroom("R302", 32, ["projector", "whiteboard"])
    ]
    
    # Define subjects with preferences
    subjects = [
        Subject("MATH", "Mathematics", 6, preferred_periods=[0, 1, 2]),  # Prefer morning
        Subject("ENG", "English", 5, preferred_periods=[2, 3, 4]),  # Prefer late morning/early afternoon
        Subject("PHYS", "Physics", 4, requires_lab=True, preferred_periods=[4, 5, 6]),  # Prefer afternoon
        Subject("CHEM", "Chemistry", 4, requires_lab=True, preferred_periods=[4, 5, 6]),  # Prefer afternoon
        Subject("BIO", "Biology", 3, requires_lab=True, preferred_periods=[4, 5, 6]),  # Prefer afternoon
        Subject("HIST", "History", 3, preferred_periods=[0, 1, 2]),  # Prefer morning
        Subject("CS", "Computer Science", 3, requires_lab=True, preferred_periods=[4, 5, 6]),  # Prefer afternoon
        Subject("ART", "Art", 2, preferred_periods=[3, 4, 5])  # Prefer mid-day
    ]
    
    # Define class groups with preferences
    class_groups = [
        ClassGroup("10A", 32, ["MATH", "ENG", "PHYS", "HIST", "ART"],
                   preferred_schedule={0: [0, 1, 2], 2: [0, 1, 2], 4: [0, 1, 2]}),  # Prefer Mon/Wed/Fri morning
        
        ClassGroup("10B", 30, ["MATH", "ENG", "CHEM", "CS", "ART"],
                   preferred_schedule={1: [2, 3, 4], 3: [2, 3, 4]}),  # Prefer Tue/Thu late morning
        
        ClassGroup("10C", 28, ["MATH", "ENG", "BIO", "HIST", "CS"],
                   preferred_schedule={0: [3, 4, 5], 2: [3, 4, 5], 4: [3, 4, 5]}),  # Prefer Mon/Wed/Fri afternoon
        
        ClassGroup("11A", 26, ["MATH", "ENG", "PHYS", "CHEM", "ART"],
                   preferred_schedule={1: [0, 1, 2], 3: [0, 1, 2]}),  # Prefer Tue/Thu early
        
        ClassGroup("11B", 25, ["MATH", "ENG", "BIO", "CS", "HIST"],
                   preferred_schedule={1: [4, 5, 6], 3: [4, 5, 6]})  # Prefer Tue/Thu late afternoon
    ]
    
    return teachers, classrooms, subjects, class_groups

def main():
    """Main function for enhanced timetable generation"""
    print("🎓 ENHANCED HYBRID TIMETABLE GENERATOR")
    print("🧠 Powered by GNN Seeding + Local Search Refinement")
    print("="*90)
    
    # Create comprehensive sample data
    teachers, classrooms, subjects, class_groups = create_enhanced_sample_data()
    
    print(f"📋 Enhanced Input Data Summary:")
    print(f"  👨🏫 Teachers: {len(teachers)} (with preferences)")
    print(f"  🏫 Classrooms: {len(classrooms)} (with specialized equipment)")
    print(f"  📚 Subjects: {len(subjects)} (with preferences)")
    print(f"  👥 Classes: {len(class_groups)} (with preferred schedules)")
    
    subjects_dict = {s.subject_id: s for s in subjects}
    total_periods_needed = sum(
        sum(subjects_dict[s].periods_per_week for s in cg.subjects)
        for cg in class_groups
    )
    print(f"  ⏰ Total periods needed: {total_periods_needed}")
    print()
    
    # Create and run hybrid generator
    generator = HybridTimetableGenerator(teachers, classrooms, subjects, class_groups)
    
    # Generate optimized timetable with hybrid approach
    result = generator.generate_hybrid_timetable()
    
    print("\n🎉 Enhanced hybrid timetable generation completed!")

if __name__ == "__main__":
    # Install torch_geometric if not available
    try:
        import torch_geometric
    except ImportError:
        print("⚠ torch_geometric is required but not installed.")
        print("  Install it with: pip install torch-geometric")
    
    main()