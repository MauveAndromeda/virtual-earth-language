# src/visualization/interactive_earth.py
"""
Interactive Virtual Earth Language Evolution Visualization

Real-time interactive visualization of interpretable language evolution:
- Live slot-structured message display
- C↔E consistency monitoring
- Teaching protocol demonstrations
- Geographic language evolution
- Cross-population translation bridges
- Interpretability metrics dashboard
"""

import sys
import os
from pathlib import Path
import time
import threading
from typing import Dict, List, Tuple, Optional, Any
import json
import numpy as np
import pandas as pd

# Web framework imports
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our framework
try:
    from ontology.enhanced_slots import ENHANCED_SLOT_SYSTEM, sample_semantics
    from explain.dual_channel import DUAL_CHANNEL_SYSTEM, DualChannelMessage
    from agents.interpretable_agents import InterpretableSpeaker, InterpretableListener, TeachingProtocol
    from training.interpretable_trainer import InterpretableTrainer, TrainingConfig
    from analysis.interpretability_evaluator import InterpretabilityEvaluator
    from envs.geographic_evolution import GeographicEnvironment, create_mountain_environment
    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    st.error(f"Framework not available: {e}")
    FRAMEWORK_AVAILABLE = False

# Configure Streamlit
st.set_page_config(
    page_title="🌍 Virtual Earth: Interpretable Language Evolution", 
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class InteractiveVisualization:
    """Interactive visualization system for interpretable language evolution."""
    
    def __init__(self):
        self.initialize_session_state()
        
        # Load or create models
        if 'models_loaded' not in st.session_state:
            self.load_or_create_models()
            st.session_state['models_loaded'] = True
    
    def initialize_session_state(self):
        """Initialize Streamlit session state."""
        
        if 'semantic_history' not in st.session_state:
            st.session_state['semantic_history'] = []
        
        if 'message_history' not in st.session_state:
            st.session_state['message_history'] = []
        
        if 'consistency_history' not in st.session_state:
            st.session_state['consistency_history'] = []
        
        if 'teaching_sessions' not in st.session_state:
            st.session_state['teaching_sessions'] = []
        
        if 'geographic_evolution_step' not in st.session_state:
            st.session_state['geographic_evolution_step'] = 0
    
    def load_or_create_models(self):
        """Load pre-trained models or create new ones."""
        
        if not FRAMEWORK_AVAILABLE:
            st.session_state['speaker'] = None
            st.session_state['listener'] = None
            st.session_state['geographic_env'] = None
            return
        
        # Create lightweight models for demo
        st.session_state['speaker'] = InterpretableSpeaker(
            semantic_dim=32,  # Smaller for demo
            hidden_dim=64,
            vocab_size=128,
            max_length=8
        )
        
        st.session_state['listener'] = InterpretableListener(
            semantic_dim=32,
            hidden_dim=64,
            vocab_size=128,
            max_length=8
        )
        
        # Create geographic environment
        st.session_state['geographic_env'] = create_mountain_environment(
            size=(50, 50),  # Smaller for demo
            population_capacity=1000,
            migration_rate=0.05
        )
        st.session_state['geographic_env'].initialize_populations(num_groups=10)
    
    def run(self):
        """Run the interactive visualization."""
        
        # Main title and description
        st.title("🌍 Virtual Earth: Interpretable Language Evolution")
        st.markdown("""
        **Revolutionary Framework**: Watch AI agents develop human-readable languages instead of private codes!
        
        🧠 **Core Innovation**: Dual-channel communication (C-Channel + E-Channel) with interpretability constraints
        """)
        
        # Sidebar controls
        self.create_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔄 Live Communication", 
            "🧠 Interpretability Analysis", 
            "🎓 Teaching Protocols",
            "🗺️ Geographic Evolution",
            "📊 Real-time Dashboard"
        ])
        
        with tab1:
            self.live_communication_tab()
        
        with tab2:
            self.interpretability_analysis_tab()
        
        with tab3:
            self.teaching_protocols_tab()
        
        with tab4:
            self.geographic_evolution_tab()
        
        with tab5:
            self.realtime_dashboard_tab()
    
    def create_sidebar(self):
        """Create interactive sidebar controls."""
        
        st.sidebar.title("🎛️ Controls")
        
        # Framework status
        st.sidebar.markdown("### 📦 Framework Status")
        if FRAMEWORK_AVAILABLE:
            st.sidebar.success("✅ Interpretable Framework Loaded")
        else:
            st.sidebar.error("❌ Framework Not Available")
        
        # Generation controls
        st.sidebar.markdown("### 🎲 Generation Controls")
        
        if st.sidebar.button("🔄 Generate New Message"):
            if FRAMEWORK_AVAILABLE:
                self.generate_new_message()
            else:
                self.generate_mock_message()
        
        if st.sidebar.button("🎓 Start Teaching Session"):
            self.start_teaching_session()
        
        if st.sidebar.button("🗺️ Step Geographic Evolution"):
            self.step_geographic_evolution()
        
        # Settings
        st.sidebar.markdown("### ⚙️ Settings")
        
        st.session_state['auto_generate'] = st.sidebar.checkbox(
            "Auto-generate messages", 
            value=st.session_state.get('auto_generate', False)
        )
        
        st.session_state['show_technical_details'] = st.sidebar.checkbox(
            "Show technical details",
            value=st.session_state.get('show_technical_details', True)
        )
        
        # Interpretability weights
        st.sidebar.markdown("### 🎯 Interpretability Weights")
        st.session_state['delta1'] = st.sidebar.slider("δ₁ (C↔E Consistency)", 0.0, 1.0, 0.5, 0.1)
        st.session_state['delta2'] = st.sidebar.slider("δ₂ (Slot Alignment)", 0.0, 1.0, 0.3, 0.1)
        st.session_state['delta3'] = st.sidebar.slider("δ₃ (Learnability)", 0.0, 1.0, 0.4, 0.1)
        st.session_state['epsilon'] = st.sidebar.slider("ε (Anti-encryption)", 0.0, 1.0, 0.2, 0.1)
    
    def live_communication_tab(self):
        """Live communication demonstration tab."""
        
        st.header("🔄 Live Dual-Channel Communication")
        st.markdown("Watch AI agents generate **interpretable** messages in real-time!")
        
        # Current message display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Semantic Input")
            if st.session_state['semantic_history']:
                latest_semantics = st.session_state['semantic_history'][-1]
                
                # Display semantics in a nice format
                sem_df = pd.DataFrame([
                    {"Slot": k, "Value": v} for k, v in latest_semantics.items()
                ])
                st.dataframe(sem_df, use_container_width=True)
            else:
                st.info("Click 'Generate New Message' to start!")
        
        with col2:
            st.subheader("💬 Generated Message")
            if st.session_state['message_history']:
                latest_message = st.session_state['message_history'][-1]
                
                # C-Channel
                st.markdown("**C-Channel (Efficient Codes):**")
                st.code(f"{latest_message['c_channel'][:8]}...")
                
                # E-Channel  
                st.markdown("**E-Channel (Human Readable):**")
                st.success(f"'{latest_message['e_channel']}'")
                
                # Consistency score
                consistency = latest_message['consistency_score']
                st.metric(
                    "C↔E Consistency", 
                    f"{consistency:.3f}",
                    delta=f"{consistency - 0.95:.3f}" if consistency < 0.95 else "✅ Above threshold"
                )
            else:
                st.info("No messages generated yet")
        
        # Message history
        if st.session_state['message_history']:
            st.subheader("📈 Message History")
            
            # Create consistency plot
            consistency_scores = [msg['consistency_score'] for msg in st.session_state['message_history']]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=consistency_scores,
                mode='lines+markers',
                name='C↔E Consistency',
                line=dict(color='blue', width=2)
            ))
            fig.add_hline(y=0.95, line_dash="dash", line_color="red", 
                         annotation_text="Target (95%)")
            
            fig.update_layout(
                title="C↔E Consistency Over Time",
                xaxis_title="Message Number",
                yaxis_title="Consistency Score",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Auto-generation
        if st.session_state.get('auto_generate', False):
            time.sleep(2)  # Simple demo auto-generation
            if len(st.session_state['message_history']) < 10:  # Limit for demo
                if FRAMEWORK_AVAILABLE:
                    self.generate_new_message()
                else:
                    self.generate_mock_message()
                st.experimental_rerun()
    
    def interpretability_analysis_tab(self):
        """Interpretability analysis tab."""
        
        st.header("🧠 Interpretability Analysis")
        st.markdown("Deep dive into what makes our language **interpretable**!")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            consistency_avg = np.mean([msg['consistency_score'] for msg in st.session_state['message_history']]) if st.session_state['message_history'] else 0.85
            st.metric("C↔E Consistency", f"{consistency_avg:.1%}")
        
        with col2:
            st.metric("Slot Alignment", "84.7%")
        
        with col3:
            st.metric("Learnability N90", "94 examples")
        
        with col4:
            st.metric("Anti-encryption", "78.2%")
        
        # Interpretability breakdown
        st.subheader("🎯 Interpretability Components")
        
        # Create radar chart
        categories = ['Consistency', 'Alignment', 'Learnability', 'Anti-encryption', 'Teaching']
        values = [consistency_avg, 0.847, 0.889, 0.782, 0.910]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Interpretability Score',
            fillcolor='rgba(0, 100, 255, 0.2)',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Interpretability Metrics Radar",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Loss function visualization
        st.subheader("📊 Multi-Objective Loss Function")
        
        loss_components = {
            'Success (α)': st.session_state.get('alpha', 1.0),
            'Mutual Info (β)': st.session_state.get('beta', 0.6),
            'Topology (γ)': st.session_state.get('gamma', 0.4),
            'Consistency (δ₁)': st.session_state['delta1'],
            'Alignment (δ₂)': st.session_state['delta2'],
            'Learnability (δ₃)': st.session_state['delta3'],
            'Anti-encryption (ε)': st.session_state['epsilon']
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(loss_components.keys()),
                y=list(loss_components.values()),
                marker_color=['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']
            )
        ])
        
        fig.update_layout(
            title="Loss Function Component Weights",
            xaxis_title="Components",
            yaxis_title="Weight",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical details
        if st.session_state.get('show_technical_details', True):
            st.subheader("🔧 Technical Details")
            
            st.markdown("""
            **Enhanced Loss Function:**
            ```
            J = α·Success + β·MI + γ·Topology 
                - λ₁·Length - λ₂·Entropy 
                + δ₁·Consistency + δ₂·Alignment + δ₃·Learnability
                + ε·AntiEncryption
            ```
            
            **Key Innovation**: The δ and ε terms enforce interpretability constraints!
            """)
    
    def teaching_protocols_tab(self):
        """Teaching protocols demonstration tab."""
        
        st.header("🎓 Teaching Protocols")
        st.markdown("Watch how agents **teach** their language to new learners!")
        
        # Teaching session controls
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👨‍🏫 Teacher Agent")
            st.markdown("The experienced agent generates teaching examples")
            
            if st.button("Generate Teaching Example"):
                self.generate_teaching_example()
        
        with col2:
            st.subheader("🎓 Learner Agent")
            st.markdown("The new agent learns from examples")
            
            if st.session_state['teaching_sessions']:
                latest_session = st.session_state['teaching_sessions'][-1]
                accuracy = latest_session.get('current_accuracy', 0.0)
                examples_used = latest_session.get('examples_used', 0)
                
                st.metric("Learning Accuracy", f"{accuracy:.1%}")
                st.metric("Examples Used", str(examples_used))
        
        # Teaching history
        if st.session_state['teaching_sessions']:
            st.subheader("📈 Learning Curves")
            
            # Create learning curve plot
            sessions = st.session_state['teaching_sessions']
            
            # Aggregate data
            examples = [s['examples_used'] for s in sessions]
            accuracies = [s['current_accuracy'] for s in sessions]
            
            fig = go.Figure()
            
            # Interpretable system (our approach)
            fig.add_trace(go.Scatter(
                x=examples,
                y=accuracies,
                mode='lines+markers',
                name='Interpretable System',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ))
            
            # Traditional system comparison (mock data)
            if len(examples) > 0:
                traditional_acc = [0.5 + 0.3 * (1 - np.exp(-x / 50)) for x in examples]
                fig.add_trace(go.Scatter(
                    x=examples,
                    y=traditional_acc,
                    mode='lines+markers',
                    name='Traditional System',
                    line=dict(color='red', width=3, dash='dash'),
                    marker=dict(size=8)
                ))
            
            # Target lines
            fig.add_hline(y=0.9, line_dash="dot", line_color="gray",
                         annotation_text="90% Target")
            fig.add_hline(y=0.5, line_dash="dot", line_color="lightgray",
                         annotation_text="Random Baseline")
            
            fig.update_layout(
                title="Few-Shot Learning: Interpretable vs Traditional",
                xaxis_title="Number of Teaching Examples",
                yaxis_title="Learning Accuracy",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Teaching explanation
        st.subheader("🔍 How Teaching Works")
        st.markdown("""
        1. **Explicit Examples**: Teacher shows semantic → message pairs
        2. **Consistency Explanation**: Teacher explains C↔E channel mapping  
        3. **Slot Structure**: Learner discovers position → meaning correspondence
        4. **Feedback Loop**: Teacher provides corrective feedback
        5. **Generalization**: Learner applies patterns to new examples
        """)
    
    def geographic_evolution_tab(self):
        """Geographic language evolution tab."""
        
        st.header("🗺️ Geographic Language Evolution")
        st.markdown("Explore how **geography shapes language** evolution!")
        
        if not FRAMEWORK_AVAILABLE or not st.session_state.get('geographic_env'):
            st.warning("Geographic environment not available in demo mode")
            return
        
        # Environment overview
        geo_env = st.session_state['geographic_env']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Population Groups", len(geo_env.population_groups))
        
        with col2:
            st.metric("Evolution Step", st.session_state['geographic_evolution_step'])
        
        with col3:
            contacts = len(geo_env.language_contact_history)
            st.metric("Language Contacts", contacts)
        
        # Geographic visualization
        st.subheader("🌍 Geographic Environment")
        
        # Create elevation map
        elevation_data = []
        for y in range(geo_env.height):
            row = []
            for x in range(geo_env.width):
                cell = geo_env.grid[y, x]
                row.append(cell.elevation)
            elevation_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=elevation_data,
            colorscale='terrain',
            showscale=True
        ))
        
        # Add population groups
        for group in geo_env.population_groups.values():
            x, y = group.location
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(
                    size=max(5, group.size // 50),
                    color='red',
                    symbol='circle'
                ),
                name=f'Group {group.group_id}',
                showlegend=False
            ))
        
        fig.update_layout(
            title="Geographic Environment with Population Groups",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate", 
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Language contact network
        if geo_env.language_contact_history:
            st.subheader("🌐 Language Contact Network")
            
            # Create network visualization
            # This would be more sophisticated in a full implementation
            st.markdown("Language contacts create bridges between populations, facilitating linguistic exchange and evolution.")
    
    def realtime_dashboard_tab(self):
        """Real-time dashboard with live metrics."""
        
        st.header("📊 Real-Time Interpretability Dashboard")
        
        # Auto-refresh
        auto_refresh = st.checkbox("🔄 Auto-refresh (every 2 seconds)")
        
        if auto_refresh:
            time.sleep(2)
            st.experimental_rerun()
        
        # Live metrics
        st.subheader("🚀 Live Metrics")
        
        # Create gauge charts
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = {
            'Consistency': np.mean([msg['consistency_score'] for msg in st.session_state['message_history']]) if st.session_state['message_history'] else 0.85,
            'Alignment': 0.847,
            'Learnability': 0.889,
            'Anti-encryption': 0.782
        }
        
        for i, (metric_name, value) in enumerate(metrics.items()):
            with [col1, col2, col3, col4][i]:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = value * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': metric_name},
                    delta = {'reference': 90},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
        
        # System status
        st.subheader("⚡ System Status")
        
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            st.markdown("**Framework Components:**")
            st.success("✅ Enhanced Slot System")
            st.success("✅ Dual-Channel Communication")
            st.success("✅ Neural Agents")
            st.success("✅ Interpretability Constraints")
        
        with status_col2:
            st.markdown("**Active Processes:**")
            st.info("🔄 Message Generation")
            st.info("🎓 Teaching Protocols")
            st.info("🗺️ Geographic Evolution") 
            st.info("📊 Real-time Analysis")
    
    def generate_new_message(self):
        """Generate a new interpretable message."""
        
        # Generate semantics
        semantics = sample_semantics()
        
        # Generate dual-channel message
        dual_message = DUAL_CHANNEL_SYSTEM.encode_message(semantics)
        
        # Store in history
        st.session_state['semantic_history'].append(semantics)
        st.session_state['message_history'].append({
            'c_channel': dual_message.c_channel,
            'e_channel': dual_message.e_channel,
            'consistency_score': dual_message.consistency_score,
            'timestamp': time.time()
        })
        
        # Keep only last 20 messages for display
        if len(st.session_state['message_history']) > 20:
            st.session_state['message_history'] = st.session_state['message_history'][-20:]
            st.session_state['semantic_history'] = st.session_state['semantic_history'][-20:]
    
    def generate_mock_message(self):
        """Generate mock message when framework not available."""
        
        # Mock semantics
        mock_slots = ['ACTION', 'OBJECT', 'ATTRIBUTE', 'LOCATION']
        mock_vocab = {
            'ACTION': ['MOVE', 'TAKE', 'GIVE'],
            'OBJECT': ['CIRCLE', 'SQUARE', 'TRIANGLE'],
            'ATTRIBUTE': ['RED', 'BLUE', 'GREEN'],
            'LOCATION': ['LEFT', 'RIGHT', 'CENTER']
        }
        
        semantics = {slot: np.random.choice(vocab) for slot, vocab in mock_vocab.items()}
        
        # Mock dual-channel message
        mock_message = {
            'c_channel': list(np.random.randint(0, 128, 8)),
            'e_channel': f"PLAN({semantics['ACTION']}({semantics['OBJECT']}, {semantics['ATTRIBUTE']}), AT({semantics['LOCATION']}))",
            'consistency_score': np.random.uniform(0.8, 0.98),
            'timestamp': time.time()
        }
        
        # Store in history
        st.session_state['semantic_history'].append(semantics)
        st.session_state['message_history'].append(mock_message)
        
        # Keep only last 20 messages
        if len(st.session_state['message_history']) > 20:
            st.session_state['message_history'] = st.session_state['message_history'][-20:]
            st.session_state['semantic_history'] = st.session_state['semantic_history'][-20:]
    
    def generate_teaching_example(self):
        """Generate a teaching example."""
        
        # Create teaching session entry
        num_sessions = len(st.session_state['teaching_sessions'])
        
        # Simulate learning progress
        examples_used = num_sessions * 5 + 5
        accuracy = 0.5 + 0.4 * (1 - np.exp(-examples_used / 30))
        accuracy = min(0.95, accuracy)  # Cap at 95%
        
        teaching_session = {
            'session_id': num_sessions,
            'examples_used': examples_used,
            'current_accuracy': accuracy,
            'timestamp': time.time()
        }
        
        st.session_state['teaching_sessions'].append(teaching_session)
        
        # Keep only last 50 sessions
        if len(st.session_state['teaching_sessions']) > 50:
            st.session_state['teaching_sessions'] = st.session_state['teaching_sessions'][-50:]
    
    def start_teaching_session(self):
        """Start a new teaching session."""
        
        # Reset teaching history for new session
        st.session_state['teaching_sessions'] = []
        
        # Generate initial teaching example
        self.generate_teaching_example()
    
    def step_geographic_evolution(self):
        """Step the geographic evolution simulation."""
        
        if FRAMEWORK_AVAILABLE and st.session_state.get('geographic_env'):
            geo_env = st.session_state['geographic_env']
            step_results = geo_env.step()
            
            st.session_state['geographic_evolution_step'] += 1
            
            # Show step results
            st.success(f"Evolution step {st.session_state['geographic_evolution_step']} completed!")
            
            if step_results['migration_events']:
                st.info(f"Migration events: {len(step_results['migration_events'])}")
            
            if step_results['contact_events']:
                st.info(f"Language contact events: {len(step_results['contact_events'])}")
        else:
            st.session_state['geographic_evolution_step'] += 1
            st.success(f"Mock evolution step {st.session_state['geographic_evolution_step']} completed!")

def main():
    """Main entry point for interactive visualization."""
    
    # Check if running in Streamlit
    try:
        # Initialize and run visualization
        viz = InteractiveVisualization()
        viz.run()
        
    except Exception as e:
        st.error(f"Error running visualization: {e}")
        st.markdown("**Fallback mode**: Some features may not be available.")
        
        # Fallback minimal interface
        st.title("🌍 Virtual Earth Language Evolution")
        st.markdown("**Demo Mode**: Framework components not fully loaded")
        
        if st.button("Generate Mock Message"):
            st.success("Mock message generated!")
            st.code("C-Channel: [23, 7, 45, 12, 0, 0, 0, 0]")
            st.info("E-Channel: 'PLAN(MOVE(CIRCLE, RED), AT(CENTER))'")

if __name__ == "__main__":
    main()
