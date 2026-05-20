
"""
Productivity Intelligence Agent - Streamlit Web Dashboard
Run with: streamlit run productivity_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# Page config
st.set_page_config(
    page_title="Productivity Intelligence Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; }
    .sub-header { font-size: 1.2rem; color: #666; }
    .metric-card { background: #f0f2f6; padding: 1rem; border-radius: 10px; }
    .peak-badge { background: #ff6b6b; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; }
    .good-badge { background: #4ecdc4; color: white; padding: 4px 12px; border-radius: 20px; }
    .warning-badge { background: #ffd93d; color: black; padding: 4px 12px; border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    # Import the agent class (assumes productivity_agent.py is in same directory)
    try:
        from productivity_agent import ProductivityIntelligenceAgent
        st.session_state.agent = ProductivityIntelligenceAgent(user_name="Engineer")

        # Load demo data if available
        if os.path.exists("my_productivity_data.json"):
            st.session_state.agent.import_data("my_productivity_data.json")
    except ImportError:
        st.error("⚠️ productivity_agent.py not found. Please place it in the same directory.")
        st.stop()

agent = st.session_state.agent

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("## 🧠 Agent Controls")

    # Add Task Form
    with st.expander("➕ Add New Task", expanded=False):
        with st.form("add_task_form"):
            task_name = st.text_input("Task Name", placeholder="e.g., Bridge deck analysis")
            category = st.selectbox("Category", ["Structural Design", "Analysis", "Detailing", "Materials", "Other"])
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
            end_date = st.date_input("End Date", datetime.now())

            col1, col2 = st.columns(2)
            with col1:
                complexity = st.slider("Complexity", 1, 10, 5)
                satisfaction = st.slider("Satisfaction", 1, 10, 7)
            with col2:
                output_quality = st.slider("Output Quality", 1, 10, 7)
                collaboration = st.slider("Collaboration", 1, 10, 5)

            notes = st.text_area("Notes", placeholder="Any additional context...")

            submitted = st.form_submit_button("🚀 Add Task & Train Models")

            if submitted and task_name:
                agent.add_task(
                    task_name=task_name,
                    category=category,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    complexity=complexity,
                    satisfaction=satisfaction,
                    output_quality=output_quality,
                    collaboration=collaboration,
                    notes=notes
                )
                st.success(f"✅ Added: {task_name}")
                st.rerun()

    # Predict Task
    with st.expander("🔮 Predict Next Task", expanded=False):
        with st.form("predict_form"):
            pred_cat = st.selectbox("Category", ["Structural Design", "Analysis", "Detailing", "Materials"], key="pred_cat")
            pred_comp = st.slider("Complexity", 1, 10, 7, key="pred_comp")
            pred_collab = st.slider("Collaboration", 1, 10, 6, key="pred_collab")

            if st.form_submit_button("Predict Productivity"):
                result = agent.predict_next_task(pred_cat, pred_comp, pred_collab)
                st.metric("Predicted Score", f"{result['predicted_score']:.2f}")
                st.info(result['recommendation'])
                st.caption(f"Confidence: {result['confidence']}")

    # Optimal Schedule
    with st.expander("📅 Optimal Schedule", expanded=False):
        st.caption("Enter up to 5 upcoming tasks")
        upcoming = []
        for i in range(5):
            cols = st.columns([2, 1, 1])
            with cols[0]:
                t_name = st.text_input(f"Task {i+1}", key=f"sched_name_{i}", placeholder=f"Task {i+1} name")
            with cols[1]:
                t_cat = st.selectbox(f"Cat", ["Structural Design", "Analysis", "Detailing", "Materials"], key=f"sched_cat_{i}")
            with cols[2]:
                t_comp = st.number_input(f"C", 1, 10, 5, key=f"sched_comp_{i}")

            if t_name:
                upcoming.append({"task": t_name, "category": t_cat, "complexity": t_comp, "collaboration": 5})

        if st.button("🎯 Optimize Schedule") and upcoming:
            schedule = agent.get_optimal_schedule(upcoming)
            for i, task in enumerate(schedule, 1):
                with st.container():
                    st.markdown(f"**{i}. {task['task']}** — Score: `{task['predicted_score']:.2f}`")
                    st.caption(task['schedule_advice'])

    st.markdown("---")
    if st.button("💾 Export Data"):
        agent.export_data("my_productivity_data.json")
        st.success("Data exported!")

    if st.button("🩺 Check Burnout Risk"):
        risk = agent.detect_burnout_risk()
        if risk['risk'] == 'HIGH':
            st.error(f"🔴 {risk['risk']} RISK: {risk['advice']}")
        elif risk['risk'] == 'MEDIUM':
            st.warning(f"🟡 {risk['risk']} RISK: {risk['advice']}")
        else:
            st.success(f"🟢 {risk['risk']} RISK: {risk['advice']}")

# ============================================
# MAIN DASHBOARD
# ============================================
st.markdown('<p class="main-header">🧠 Productivity Intelligence Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your Personal AI Work Coach — Tracking, Predicting, Optimizing</p>', unsafe_allow_html=True)

# Top Metrics Row
if len(agent.data) > 0:
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Tasks", len(agent.data))
    with col2:
        st.metric("Days Tracked", int(agent.state['total_days']))
    with col3:
        st.metric("Peak Tasks", int(agent.state['peak_count']))
    with col4:
        avg_score = agent.data['productivity_score'].mean()
        st.metric("Avg Productivity", f"{avg_score:.2f}")
    with col5:
        trend = "📈" if agent.state['trend_slope'] > 0.05 else "📉" if agent.state['trend_slope'] < -0.05 else "➡️"
        st.metric("Trend", f"{trend} {agent.state['trend_slope']:+.2f}")

    st.markdown("---")

    # Two column layout for charts
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("📈 Productivity Timeline")

        df_plot = agent.data.copy()
        df_plot['start_date_str'] = df_plot['start_date'].dt.strftime('%Y-%m-%d')

        fig = px.scatter(
            df_plot, 
            x='start_date', 
            y='productivity_score',
            size='complexity',
            color='category',
            hover_data=['task', 'efficiency', 'velocity'],
            title="Tasks over Time (bubble size = complexity)"
        )

        # Highlight peaks
        peaks_df = df_plot[df_plot['is_peak'] == 1]
        if len(peaks_df) > 0:
            fig.add_trace(go.Scatter(
                x=peaks_df['start_date'],
                y=peaks_df['productivity_score'],
                mode='markers',
                marker=dict(size=20, color='yellow', symbol='star', line=dict(width=2, color='red')),
                name='🔥 Peak Performance',
                hovertemplate='<b>PEAK</b><br>%{text}<extra></extra>',
                text=peaks_df['task']
            ))

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.subheader("💪 Skill Profile")
        profile = agent.get_skill_profile()
        if 'ranking' in profile:
            skill_df = pd.DataFrame({
                'Category': list(profile['ranking'].keys()),
                'Score': list(profile['ranking'].values())
            }).sort_values('Score', ascending=True)

            fig_skills = px.bar(
                skill_df, 
                x='Score', 
                y='Category', 
                orientation='h',
                color='Score',
                color_continuous_scale='RdYlGn',
                range_x=[0, 10]
            )
            fig_skills.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_skills, use_container_width=True)

    # Second row
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🔥 Peak Performance Report")
        peak_report = agent.get_peak_performance_report()
        if 'tasks' in peak_report:
            st.markdown(f"**Peak Tasks Detected:** {peak_report['total_peak_tasks']}")
            st.markdown(f"**Pattern:** {peak_report['pattern']}")
            st.markdown("**Peak Tasks:**")
            for task in peak_report['tasks']:
                st.markdown(f"- <span class='peak-badge'>{task}</span>", unsafe_allow_html=True)
        else:
            st.info("Add more tasks to detect peak performance patterns.")

    with col_right:
        st.subheader("🎯 Recommendations")
        report = agent.generate_full_report()
        for rec in report['recommendations'][:5]:
            if 'SUPERSTRENGTH' in rec or 'Double down' in rec:
                st.success(rec)
            elif 'GROWTH' in rec or 'Invest' in rec or 'burnout' in rec.lower():
                st.warning(rec)
            elif 'WEB' in rec or 'BENCHMARK' in rec:
                st.info(rec)
            else:
                st.write(rec)

    # Third row - Detailed data table
    st.markdown("---")
    st.subheader("📋 All Tasks Data")

    display_df = agent.data[['task', 'category', 'start_date', 'days', 'complexity', 
                             'productivity_score', 'efficiency', 'is_peak', 'cluster_label']].copy()
    display_df['start_date'] = display_df['start_date'].dt.strftime('%Y-%m-%d')
    display_df['productivity_score'] = display_df['productivity_score'].round(2)
    display_df['efficiency'] = display_df['efficiency'].round(2)
    display_df['is_peak'] = display_df['is_peak'].map({1: '🔥 PEAK', 0: '-'})

    st.dataframe(display_df, use_container_width=True, height=300)

    # Raw data export
    with st.expander("📤 View Raw JSON"):
        st.json(report)

else:
    st.info("👋 Welcome! Add your first task using the sidebar to see your dashboard.")
    st.markdown("""
    ### Quick Start:
    1. Click **➕ Add New Task** in the sidebar
    2. Enter a task you recently completed
    3. Rate complexity, satisfaction, quality, and collaboration
    4. The agent will immediately train models and show insights
    5. Add at least 3 tasks for full ML features
    """)
