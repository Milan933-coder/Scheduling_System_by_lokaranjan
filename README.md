GNN-MCTS Timetable Generator
A sophisticated, AlphaZero-inspired AI agent that solves the complex University Course Timetabling Problem (UCTP). This project uses a hybrid approach combining Graph Neural Networks (GNN), Monte Carlo Tree Search (MCTS), and Local Search to find optimal solutions.

🚀 Overview
The University Course Timetabling Problem (UCTP) is a classic NP-hard optimization problem. The goal is to assign a set of events (classes, subjects) to a limited number of resources (teachers, rooms) and time slots, all while satisfying a large set of constraints.

This project tackles the problem by framing it as a "game" that an AI agent can learn to play. The agent's goal is to "play" the game of scheduling classes, making optimal "moves" (assignments) to achieve the highest "score" (a high-quality, conflict-free timetable).

This system is inspired by AlphaZero, using a Graph Neural Network (GNN) as the "brain" to guide a powerful Monte Carlo Tree Search (MCTS) "searcher."

🧠 Core Architecture
The system is built on two primary components: the solver and the trainer.

Graph Representation (EnhancedGraphTimetableState): The entire problem—including teachers, subjects, classrooms, class groups, and all 40 time slots—is modeled as a large, heterogeneous graph. Nodes (e.g., Teacher, TimeSlot) and edges (e.g., Teacher "is available at" TimeSlot) have rich features describing their properties and relationships.

GNN "Brain" (EnhancedGNNPolicyValueNetwork): A Graph Attention Network (GATConv) acts as the agent's "intuition". It is trained to do two things:

Policy Network: Predict the best next action (e.g., "Schedule Math for Class 10A in Room 101 at 9 AM").

Value Network: Predict the final quality score of the entire schedule based on the current partial assignments.

MCTS "Searcher" (EnhancedNeuralMCTS): A Monte Carlo Tree Search algorithm uses the GNN's policy and value predictions to intelligently explore the vast search space of possible timetables, focusing on the most promising "moves".

Self-Play & Training Loop (TrainingPipeline_fixed.py): A reinforcement learning pipeline, inspired by AlphaZero, trains the GNN "brain":

Generate Data: The current GNN-guided MCTS plays "games" of self-play to generate a set of timetables.

Train Network: The GNN is trained on this data. It learns to predict the MCTS's final search probabilities (policy) and the final game's outcome (value).

Repeat: This "play -> train -> improve" loop repeats, making the GNN smarter with each iteration.

Hybrid Solver (HybridTimetableGenerator): To generate the final solution, the system uses a hybrid approach:

Step 1 (Seeding): The trained MCTS agent quickly generates a "good" initial solution.

Step 2 (Refinement): A Local Search algorithm (Simulated Annealing) "polishes" this solution by making small swaps to resolve minor conflicts and improve the final fitness score.

✨ Key Features
AlphaZero-Inspired Design: Uses a self-play and MCTS-based training loop to learn a complex problem from scratch.

Graph-Based State: Represents the entire timetabling problem as a heterogeneous graph, a more natural and powerful representation than a simple grid.

Hybrid Solving: Combines the strengths of deep learning (GNN) and classical search (MCTS, Local Search) for robust optimization.

Rich Constraint Handling: The graph structure is designed to handle complex, real-world constraints:

Teacher availability and preferences.

Room capacity and equipment (e.g., 'lab', 'projector').

Subject requirements (periods per week, lab required).

Class group preferences.

Complete Training Pipeline: Includes data generation, model training, checkpointing, and evaluation.
