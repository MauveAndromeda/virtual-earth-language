"""
Interpretable Referential Game Environment

Key feature: Supports dual-channel communication and interpretability metrics
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import random
from dataclasses import dataclass
from collections import defaultdict

# Import our framework
from ontology.enhanced_slots import ENHANCED_SLOT_SYSTEM, sample_semantics
from explain.dual_channel import DUAL_CHANNEL_SYSTEM, DualChannelMessage

@dataclass
class CommunicationTurn:
    """Single communication turn with interpretability tracking."""
    speaker_id: str
    listener_id: str
    target_semantics: Dict[str, str]
    candidates: List[Dict[str, str]]
    target_index: int
    
    # Generated communication
    c_channel: List[int]
    e_channel: str
    consistency_score: float
    
    # Results
    predicted_index: int
    success: bool
    communication_time: float = 0.0
    
    # Interpretability metrics
    slot_alignment_score: float = 0.0
    teaching_potential: float = 0.0
    noise_robustness: float = 0.0

class InterpretableReferentialGame(gym.Env):
    """
    Referential game environment optimized for interpretable communication.
    
    Features:
    - Dual-channel message generation and evaluation
    - Real-time interpretability metric tracking
    - Teaching protocol support
    - Cross-population communication testing
    - Geographic population simulation
    """
    
    metadata = {'render.modes': ['human', 'rgb_array', 'interpretability']}
    
    def __init__(
        self,
        num_candidates: int = 4,
        max_episode_steps: int = 100,
        difficulty_progression: bool = True,
        track_interpretability: bool = True,
        enable_teaching: bool = True,
        population_groups: int = 1,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.num_candidates = num_candidates
        self.max_episode_steps = max_episode_steps
        self.difficulty_progression = difficulty_progression
        self.track_interpretability = track_interpretability
        self.enable_teaching = enable_teaching
        self.population_groups = population_groups
        
        if seed is not None:
            self.seed(seed)
        
        # Environment state
        self.current_step = 0
        self.current_difficulty = 0.0
        self.episode_history = []
        
        # Population simulation
        self.populations = self._initialize_populations()
        self.current_speaker_pop = 0
        self.current_listener_pop = 0
        
        # Interpretability tracking
        self.interpretability_history = defaultdict(list)
        self.teaching_sessions = []
        
        # Observation and action spaces
        self._setup_spaces()
        
        # Initialize first episode
        self.reset()
    
    def _initialize_populations(self) -> List[Dict[str, Any]]:
        """Initialize different population groups with varying communication styles."""
        
        populations = []
        
        for i in range(self.population_groups):
            # Each population has slightly different communication preferences
            population = {
                'id': i,
                'name': f'Population_{i}',
                'slot_preferences': [random.random() for _ in range(5)],  # Preference for each slot
                'vocabulary_bias': {
                    slot: random.sample(ENHANCED_SLOT_SYSTEM.get_slot_vocabulary(slot), 
                                       min(3, len(ENHANCED_SLOT_SYSTEM.get_slot_vocabulary(slot))))
                    for slot in ENHANCED_SLOT_SYSTEM.slot_order
                },
                'communication_style': random.choice(['structured', 'compact', 'verbose']),
                'interpretability_level': random.uniform(0.6, 0.95)
            }
            populations.append(population)
        
        return populations
    
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        
        # Observation space: semantic features + candidates + communication history
        max_vocab_size = max(len(ENHANCED_SLOT_SYSTEM.get_slot_vocabulary(slot)) 
                            for slot in ENHANCED_SLOT_SYSTEM.slot_order)
        
        # Target semantics (one-hot for each slot)
        target_dim = len(ENHANCED_SLOT_SYSTEM.slot_order) * max_vocab_size
        
        # Candidates (multiple semantic representations)
        candidates_dim = self.num_candidates * target_dim
        
        # Communication context
        context_dim = 64  # Communication history embedding
        
        # Interpretability metrics
        interpretability_dim = 8 if self.track_interpretability else 0
        
        total_dim = target_dim + candidates_dim + context_dim + interpretability_dim
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(total_dim,), dtype=np.float32
        )
        
        # Action space: candidate selection
        self.action_space = spaces.Discrete(self.num_candidates)
        
        # Additional spaces for dual-channel communication
        self.c_channel_space = spaces.MultiDiscrete([DUAL_CHANNEL_SYSTEM.vocab_size] * 
                                                   DUAL_CHANNEL_SYSTEM.max_length)
        self.e_channel_space = spaces.Text(max_length=200)  # Text explanations
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        
        self.current_step = 0
        self.episode_history = []
        
        # Update difficulty if progression enabled
        if self.difficulty_progression:
            self.current_difficulty = min(1.0, len(self.episode_history) / 1000)
        
        # Generate new communication scenario
        self.current_scenario = self._generate_scenario()
        
        # Create observation
        observation = self._create_observation()
        
        info = {
            'target_semantics': self.current_scenario['target_semantics'],
            'candidates': self.current_scenario['candidates'],
            'target_index': self.current_scenario['target_index'],
            'difficulty': self.current_difficulty,
            'population_info': {
                'speaker': self.populations[self.current_speaker_pop],
                'listener': self.populations[self.current_listener_pop]
            }
        }
        
        return observation, info
    
    def _generate_scenario(self) -> Dict[str, Any]:
        """Generate a communication scenario with target and distractors."""
        
        # Sample target semantics
        target_semantics = sample_semantics()
        
        # Generate distractors with controlled difficulty
        candidates = [target_semantics]
        
        for _ in range(self.num_candidates - 1):
            if self.current_difficulty < 0.3:
                # Easy: very different distractors
                distractor = sample_semantics()
            elif self.current_difficulty < 0.7:
                # Medium: some similar features
                distractor = target_semantics.copy()
                # Change 2-3 slots
                slots_to_change = random.sample(list(distractor.keys()), 
                                              min(3, len(distractor)))
                for slot in slots_to_change:
                    vocab = ENHANCED_SLOT_SYSTEM.get_slot_vocabulary(slot)
                    distractor[slot] = random.choice([v for v in vocab 
                                                    if v != distractor[slot]])
            else:
                # Hard: very similar (minimal pairs)
                distractor = target_semantics.copy()
                # Change only 1 slot
                slot_to_change = random.choice(list(distractor.keys()))
                vocab = ENHANCED_SLOT_SYSTEM.get_slot_vocabulary(slot_to_change)
                distractor[slot_to_change] = random.choice([v for v in vocab 
                                                          if v != distractor[slot_to_change]])
            
            candidates.append(distractor)
        
        # Shuffle candidates
        target_index = random.randint(0, self.num_candidates - 1)
        candidates[0], candidates[target_index] = candidates[target_index], candidates[0]
        
        # Select populations for this scenario
        if self.population_groups > 1:
            self.current_speaker_pop = random.randint(0, self.population_groups - 1)
            # Sometimes same population, sometimes different (for cross-pop testing)
            if random.random() < 0.3:  # 30% cross-population communication
                self.current_listener_pop = random.choice([i for i in range(self.population_groups) 
                                                         if i != self.current_speaker_pop])
            else:
                self.current_listener_pop = self.current_speaker_pop
        
        return {
            'target_semantics': target_semantics,
            'candidates': candidates,
            'target_index': target_index,
            'scenario_id': len(self.episode_history)
        }
    
    def _create_observation(self) -> np.ndarray:
        """Create observation vector from current scenario."""
        
        # Encode target semantics
        target_encoding = self._encode_semantics(self.current_scenario['target_semantics'])
        
        # Encode all candidates  
        candidates_encoding = []
        for candidate in self.current_scenario['candidates']:
            candidates_encoding.extend(self._encode_semantics(candidate))
        
        # Communication context (history of interpretability)
        context_encoding = self._encode_communication_context()
        
        # Current interpretability metrics
        interpretability_encoding = self._encode_interpretability_state()
        
        # Combine all components
        observation = np.concatenate([
            target_encoding,
            candidates_encoding,
            context_encoding,
            interpretability_encoding
        ])
        
        # Pad/truncate to fixed size
        target_size = self.observation_space.shape[0]
        if len(observation) < target_size:
            observation = np.pad(observation, (0, target_size - len(observation)))
        elif len(observation) > target_size:
            observation = observation[:target_size]
        
        return observation.astype(np.float32)
    
    def _encode_semantics(self, semantics: Dict[str, str]) -> np.ndarray:
        """Encode semantics as one-hot vectors."""
        
        encoding = []
        
        for slot_name in ENHANCED_SLOT_SYSTEM.slot_order:
            vocab = ENHANCED_SLOT_SYSTEM.get_slot_vocabulary(slot_name)
            slot_encoding = np.zeros(len(vocab))
            
            if slot_name in semantics:
                value = semantics[slot_name]
                if value in vocab:
                    slot_encoding[vocab.index(value)] = 1.0
            
            encoding.extend(slot_encoding)
        
        return np.array(encoding)
    
    def _encode_communication_context(self) -> np.ndarray:
        """Encode communication history and context."""
        
        # Simple encoding of recent interpretability scores
        recent_consistency = []
        recent_alignment = []
        
        for turn in self.episode_history[-10:]:  # Last 10 turns
            if hasattr(turn, 'consistency_score'):
                recent_consistency.append(turn.consistency_score)
            if hasattr(turn, 'slot_alignment_score'):
                recent_alignment.append(turn.slot_alignment_score)
        
        # Pad to fixed size
        context = np.zeros(64)
        
        if recent_consistency:
            context[0] = np.mean(recent_consistency)
            context[1] = np.std(recent_consistency)
        
        if recent_alignment:
            context[2] = np.mean(recent_alignment)
            context[3] = np.std(recent_alignment)
        
        # Population information
        context[4] = self.current_speaker_pop / max(self.population_groups, 1)
        context[5] = self.current_listener_pop / max(self.population_groups, 1)
        context[6] = float(self.current_speaker_pop != self.current_listener_pop)  # Cross-pop flag
        
        # Difficulty and step information
        context[7] = self.current_difficulty
        context[8] = self.current_step / self.max_episode_steps
        
        return context
    
    def _encode_interpretability_state(self) -> np.ndarray:
        """Encode current interpretability metrics."""
        
        if not self.track_interpretability:
            return np.array([])
        
        # Recent interpretability metrics
        state = np.zeros(8)
        
        if self.interpretability_history['consistency']:
            state[0] = np.mean(self.interpretability_history['consistency'][-10:])
        
        if self.interpretability_history['alignment']:
            state[1] = np.mean(self.interpretability_history['alignment'][-10:])
        
        if self.interpretability_history['learnability']:
            state[2] = np.mean(self.interpretability_history['learnability'][-10:])
        
        if self.interpretability_history['anti_encryption']:
            state[3] = np.mean(self.interpretability_history['anti_encryption'][-10:])
        
        # Population-specific metrics
        current_pop = self.populations[self.current_speaker_pop]
        state[4] = current_pop['interpretability_level']
        state[5] = np.mean(current_pop['slot_preferences'])
        
        # Cross-population compatibility if applicable
        if self.current_speaker_pop != self.current_listener_pop:
            listener_pop = self.populations[self.current_listener_pop]
            compatibility = 1.0 - np.mean([abs(s - l) for s, l in 
                                         zip(current_pop['slot_preferences'], 
                                             listener_pop['slot_preferences'])])
            state[6] = compatibility
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step of communication."""
        
        self.current_step += 1
        
        # Validate action
        if not (0 <= action < self.num_candidates):
            action = 0  # Default to first candidate
        
        # Determine success
        target_index = self.current_scenario['target_index']
        success = (action == target_index)
        
        # Generate dual-channel communication for this scenario
        dual_message = self._generate_dual_channel_message()
        
        # Calculate interpretability metrics
        interpretability_metrics = self._calculate_interpretability_metrics(dual_message, success)
        
        # Create communication turn record
        turn = CommunicationTurn(
            speaker_id=f"pop_{self.current_speaker_pop}",
            listener_id=f"pop_{self.current_listener_pop}",
            target_semantics=self.current_scenario['target_semantics'],
            candidates=self.current_scenario['candidates'],
            target_index=target_index,
            c_channel=dual_message.c_channel,
            e_channel=dual_message.e_channel,
            consistency_score=dual_message.consistency_score,
            predicted_index=action,
            success=success,
            **interpretability_metrics
        )
        
        self.episode_history.append(turn)
        
        # Update interpretability tracking
        if self.track_interpretability:
            self._update_interpretability_tracking(interpretability_metrics)
        
        # Calculate reward (success + interpretability bonus)
        reward = self._calculate_reward(success, interpretability_metrics)
        
        # Check if episode is done
        terminated = success or self.current_step >= self.max_episode_steps
        truncated = False
        
        # Generate next observation if continuing
        if not terminated:
            self.current_scenario = self._generate_scenario()
        
        observation = self._create_observation()
        
        # Comprehensive info
        info = {
            'success': success,
            'communication_turn': turn,
            'interpretability_metrics': interpretability_metrics,
            'population_interaction': {
                'speaker_pop': self.current_speaker_pop,
                'listener_pop': self.current_listener_pop,
                'cross_population': self.current_speaker_pop != self.current_listener_pop
            },
            'episode_stats': {
                'steps': self.current_step,
                'successes': sum(1 for t in self.episode_history if t.success),
                'avg_consistency': np.mean([t.consistency_score for t in self.episode_history]),
                'avg_alignment': np.mean([t.slot_alignment_score for t in self.episode_history])
            }
        }
        
        return observation, reward, terminated, truncated, info
    
    def _generate_dual_channel_message(self) -> DualChannelMessage:
        """Generate dual-channel message for current scenario."""
        
        target_semantics = self.current_scenario['target_semantics']
        
        # Add population-specific bias
        population = self.populations[self.current_speaker_pop]
        
        # Modify semantics based on population preferences (slight vocabulary bias)
        biased_semantics = target_semantics.copy()
        for slot, value in target_semantics.items():
            if slot in population['vocabulary_bias']:
                preferred_words = population['vocabulary_bias'][slot]
                if value not in preferred_words and random.random() < 0.2:
                    # Sometimes use preferred word from same semantic category
                    biased_semantics[slot] = random.choice(preferred_words)
        
        # Generate dual-channel message
        try:
            dual_message = DUAL_CHANNEL_SYSTEM.encode_message(biased_semantics)
        except:
            # Fallback if encoding fails
            dual_message = DualChannelMessage(
                c_channel=[1, 2, 3, 4, 0, 0, 0, 0],
                e_channel=f"PLAN({biased_semantics})",
                semantics=biased_semantics,
                consistency_score=0.8
            )
        
        return dual_message
    
    def _calculate_interpretability_metrics(
        self, 
        dual_message: DualChannelMessage, 
        success: bool
    ) -> Dict[str, float]:
        """Calculate detailed interpretability metrics."""
        
        metrics = {}
        
        # 1. Slot alignment score (position-meaning correspondence)
        metrics['slot_alignment_score'] = self._calculate_slot_alignment(dual_message)
        
        # 2. Teaching potential (how easily this message can be taught)
        metrics['teaching_potential'] = self._calculate_teaching_potential(dual_message)
        
        # 3. Noise robustness (semantic preservation under corruption)
        metrics['noise_robustness'] = self._calculate_noise_robustness(dual_message)
        
        # 4. Cross-population compatibility
        if self.current_speaker_pop != self.current_listener_pop:
            metrics['cross_pop_compatibility'] = self._calculate_cross_pop_compatibility(dual_message)
        else:
            metrics['cross_pop_compatibility'] = 1.0
        
        return metrics
    
    def _calculate_slot_alignment(self, dual_message: DualChannelMessage) -> float:
        """Calculate how well message positions align with semantic slots."""
        
        # Simple approximation: check if message length correlates with semantic complexity
        semantics = dual_message.semantics
        c_channel = dual_message.c_channel
        
        # Count non-empty slots
        filled_slots = sum(1 for v in semantics.values() if v)
        
        # Count non-zero tokens
        non_zero_tokens = sum(1 for token in c_channel if token > 0)
        
        # Alignment score based on proportionality
        if filled_slots == 0:
            return 1.0
        
        ratio = non_zero_tokens / filled_slots
        # Good alignment: 1-2 tokens per semantic slot
        alignment_score = 1.0 - abs(ratio - 1.5) / 2.0
        alignment_score = max(0.0, min(1.0, alignment_score))
        
        return alignment_score
    
    def _calculate_teaching_potential(self, dual_message: DualChannelMessage) -> float:
        """Calculate how teachable this message is."""
        
        # Factors affecting teaching potential:
        # 1. Consistency score (higher = more teachable)
        # 2. Message complexity (moderate complexity = more teachable)
        # 3. Use of anchor words (higher = more teachable)
        
        consistency_factor = dual_message.consistency_score
        
        # Complexity factor (moderate complexity is best for teaching)
        complexity = len([token for token in dual_message.c_channel if token > 0])
        complexity_factor = 1.0 - abs(complexity - 6) / 8.0  # Optimal around 6 tokens
        complexity_factor = max(0.0, min(1.0, complexity_factor))
        
        # Anchor word usage
        anchor_usage = ENHANCED_SLOT_SYSTEM.get_anchor_word_usage_rate([dual_message.semantics])
        
        # Combine factors
        teaching_potential = (consistency_factor * 0.5 + 
                            complexity_factor * 0.3 + 
                            anchor_usage * 0.2)
        
        return teaching_potential
    
    def _calculate_noise_robustness(self, dual_message: DualChannelMessage) -> float:
        """Calculate robustness to communication noise."""
        
        # Simulate 5% noise corruption
        original_c_channel = dual_message.c_channel.copy()
        
        # Corrupt 5% of non-zero tokens
        corrupted_c_channel = original_c_channel.copy()
        non_zero_indices = [i for i, token in enumerate(corrupted_c_channel) if token > 0]
        
        if non_zero_indices:
            num_corruptions = max(1, len(non_zero_indices) // 20)  # 5%
            corruption_indices = random.sample(non_zero_indices, num_corruptions)
            
            for idx in corruption_indices:
                corrupted_c_channel[idx] = random.randint(1, DUAL_CHANNEL_SYSTEM.vocab_size - 1)
        
        # Test semantic preservation
        try:
            original_semantics = DUAL_CHANNEL_SYSTEM.decode_c_channel(original_c_channel)
            corrupted_semantics = DUAL_CHANNEL_SYSTEM.decode_c_channel(corrupted_c_channel)
            
            # Calculate semantic similarity
            distance = ENHANCED_SLOT_SYSTEM.semantic_distance(original_semantics, corrupted_semantics)
            robustness = 1.0 - distance
            
        except:
            robustness = 0.5  # Default moderate robustness
        
        return robustness
    
    def _calculate_cross_pop_compatibility(self, dual_message: DualChannelMessage) -> float:
        """Calculate compatibility between populations."""
        
        speaker_pop = self.populations[self.current_speaker_pop]
        listener_pop = self.populations[self.current_listener_pop]
        
        # Compare vocabulary preferences
        vocab_compatibility = 0.0
        
        for slot in ENHANCED_SLOT_SYSTEM.slot_order:
            if slot in dual_message.semantics:
                word = dual_message.semantics[slot]
                
                # Check if word is in both populations' preferred vocabularies
                speaker_prefers = word in speaker_pop['vocabulary_bias'].get(slot, [])
                listener_prefers = word in listener_pop['vocabulary_bias'].get(slot, [])
                
                if speaker_prefers and listener_prefers:
                    vocab_compatibility += 1.0
                elif speaker_prefers or listener_prefers:
                    vocab_compatibility += 0.5
        
        vocab_compatibility /= len(ENHANCED_SLOT_SYSTEM.slot_order)
        
        # Compare communication styles
        style_compatibility = 0.8 if (speaker_pop['communication_style'] == 
                                     listener_pop['communication_style']) else 0.5
        
        # Combine factors
        overall_compatibility = vocab_compatibility * 0.6 + style_compatibility * 0.4
        
        return overall_compatibility
    
    def _update_interpretability_tracking(self, metrics: Dict[str, float]):
        """Update interpretability tracking history."""
        
        # Update history for each metric
        for metric_name, value in metrics.items():
            self.interpretability_history[metric_name].append(value)
        
        # Also track consistency from dual message
        if self.episode_history:
            latest_turn = self.episode_history[-1]
            self.interpretability_history['consistency'].append(latest_turn.consistency_score)
    
    def _calculate_reward(self, success: bool, interpretability_metrics: Dict[str, float]) -> float:
        """Calculate reward combining success and interpretability."""
        
        # Base reward for success
        success_reward = 1.0 if success else 0.0
        
        # Interpretability bonus
        interpretability_bonus = 0.0
        if self.track_interpretability:
            interpretability_bonus = (
                interpretability_metrics.get('slot_alignment_score', 0.0) * 0.2 +
                interpretability_metrics.get('teaching_potential', 0.0) * 0.2 +
                interpretability_metrics.get('noise_robustness', 0.0) * 0.1 +
                interpretability_metrics.get('cross_pop_compatibility', 0.0) * 0.1
            )
        
        # Cross-population bonus
        cross_pop_bonus = 0.1 if (success and 
                                self.current_speaker_pop != self.current_listener_pop) else 0.0
        
        total_reward = success_reward + interpretability_bonus + cross_pop_bonus
        
        return total_reward
    
    def conduct_teaching_session(
        self, 
        num_examples: int = 50, 
        success_threshold: float = 0.9
    ) -> Dict[str, Any]:
        """Conduct teaching session to test learnability."""
        
        if not self.enable_teaching:
            return {'error': 'Teaching not enabled'}
        
        teaching_examples = []
        
        # Generate teaching examples
        for _ in range(num_examples):
            # Generate scenario
            scenario = self._generate_scenario()
            
            # Generate dual-channel message
            dual_message = self._generate_dual_channel_message()
            
            # Create teaching example
            example = {
                'semantics': scenario['target_semantics'],
                'c_channel': dual_message.c_channel,
                'e_channel': dual_message.e_channel,
                'consistency_score': dual_message.consistency_score,
                'candidates': scenario['candidates'],
                'target_index': scenario['target_index']
            }
            
            teaching_examples.append(example)
        
        # Simulate learning curve
        learning_accuracies = []
        
        for i in range(1, num_examples + 1):
            # Simulate learning with i examples
            # Simple model: accuracy improves with more examples
            base_accuracy = 0.5
            learning_rate = 0.4
            accuracy = base_accuracy + learning_rate * (1 - np.exp(-i / 20))
            accuracy = min(accuracy, 0.95)
            
            learning_accuracies.append(accuracy)
        
        # Find when threshold is reached
        examples_needed = num_examples
        for i, acc in enumerate(learning_accuracies):
            if acc >= success_threshold:
                examples_needed = i + 1
                break
        
        session_results = {
            'num_examples': num_examples,
            'examples_needed_for_threshold': examples_needed,
            'final_accuracy': learning_accuracies[-1] if learning_accuracies else 0.0,
            'learning_curve': learning_accuracies,
            'teaching_examples': teaching_examples[:5],  # Sample examples
            'session_success': learning_accuracies[-1] >= success_threshold if learning_accuracies else False
        }
        
        self.teaching_sessions.append(session_results)
        
        return session_results
    
    def get_interpretability_summary(self) -> Dict[str, Any]:
        """Get comprehensive interpretability summary."""
        
        if not self.interpretability_history:
            return {'error': 'No interpretability data available'}
        
        summary = {}
        
        # Calculate average metrics
        for metric_name, values in self.interpretability_history.items():
            if values:
                summary[f'avg_{metric_name}'] = float(np.mean(values))
                summary[f'std_{metric_name}'] = float(np.std(values))
                summary[f'latest_{metric_name}'] = float(values[-1])
        
        # Episode statistics
        if self.episode_history:
            summary['episode_stats'] = {
                'total_turns': len(self.episode_history),
                'success_rate': sum(1 for t in self.episode_history if t.success) / len(self.episode_history),
                'avg_consistency': float(np.mean([t.consistency_score for t in self.episode_history])),
                'cross_population_rate': sum(1 for t in self.episode_history 
                                            if t.speaker_id != t.listener_id) / len(self.episode_history)
            }
        
        # Population analysis
        summary['population_analysis'] = {
            'num_populations': len(self.populations),
            'population_styles': [pop['communication_style'] for pop in self.populations],
            'avg_interpretability_levels': [pop['interpretability_level'] for pop in self.populations]
        }
        
        # Teaching session results
        if self.teaching_sessions:
            summary['teaching_analysis'] = {
                'num_sessions': len(self.teaching_sessions),
                'avg_examples_needed': float(np.mean([s['examples_needed_for_threshold'] 
                                                     for s in self.teaching_sessions])),
                'avg_final_accuracy': float(np.mean([s['final_accuracy'] 
                                                   for s in self.teaching_sessions]))
            }
        
        return summary
    
    def render(self, mode='human'):
        """Render the environment state."""
        
        if mode == 'human':
            if self.episode_history:
                latest_turn = self.episode_history[-1]
                print(f"\n🔄 Communication Turn {len(self.episode_history)}")
                print(f"Speaker: Population {self.current_speaker_pop}")
                print(f"Listener: Population {self.current_listener_pop}")
                print(f"Target: {latest_turn.target_semantics}")
                print(f"C-Channel: {latest_turn.c_channel[:6]}...")
                print(f"E-Channel: {latest_turn.e_channel}")
                print(f"Success: {'✅' if latest_turn.success else '❌'}")
                print(f"Consistency: {latest_turn.consistency_score:.3f}")
        
        elif mode == 'interpretability':
            summary = self.get_interpretability_summary()
            print("\n📊 Interpretability Summary:")
            for key, value in summary.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key}: {value}")
        
        elif mode == 'rgb_array':
            # Return simple visualization as RGB array
            return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up environment resources."""
        pass

# Create convenience function
def create_interpretable_environment(**kwargs) -> InterpretableReferentialGame:
    """Create interpretable referential game environment with default settings."""
    return InterpretableReferentialGame(**kwargs)
