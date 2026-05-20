
"""
🧠 PRODUCTIVITY INTELLIGENCE AGENT v2.0
A self-learning ML system for tracking, predicting, and optimizing 
structural engineering (or any) work productivity.

Author: AI-Generated for User
Date: 2026-05-20
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats
import json
import os

class ProductivityIntelligenceAgent:
    """
    Your personal AI productivity coach that:
    - Tracks work patterns
    - Detects peak performance windows
    - Predicts future productivity
    - Identifies strengths & growth areas
    - Self-upgrades via web knowledge
    - Recommends optimal task scheduling
    """

    def __init__(self, user_name="Engineer"):
        self.user_name = user_name
        self.version = "2.0"
        self.data = pd.DataFrame()
        self.models = {}
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.knowledge_base = []
        self.insights_history = []
        self.burnout_threshold = 0.3  # Efficiency drop indicator

        # Initialize with empty state
        self._init_state()

    def _init_state(self):
        """Initialize or load previous state."""
        self.state = {
            "total_tasks": 0,
            "total_days": 0,
            "peak_count": 0,
            "last_updated": datetime.now().isoformat(),
            "skill_levels": {},
            "trend_slope": 0,
            "burnout_risk": "LOW"
        }

    def add_task(self, task_name, category, start_date, end_date, 
                 complexity, satisfaction, output_quality, collaboration,
                 notes=""):
        """
        Add a new work task to the agent's memory.

        Parameters:
        -----------
        task_name : str - Description of work done
        category : str - Category (e.g., 'Structural Design', 'Analysis', 'Detailing')
        start_date : str - 'YYYY-MM-DD'
        end_date : str - 'YYYY-MM-DD'
        complexity : int - 1-10 scale
        satisfaction : int - 1-10 scale
        output_quality : int - 1-10 scale
        collaboration : int - 1-10 scale
        notes : str - Any additional notes
        """

        new_task = {
            "task": task_name,
            "category": category,
            "start_date": pd.to_datetime(start_date),
            "end_date": pd.to_datetime(end_date),
            "complexity": complexity,
            "satisfaction": satisfaction,
            "output_quality": output_quality,
            "collaboration": collaboration,
            "notes": notes,
            "days": (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1,
            "timestamp_added": datetime.now().isoformat()
        }

        # Calculate derived metrics
        new_task["productivity_score"] = (
            output_quality * 0.4 + 
            satisfaction * 0.3 + 
            (11 - complexity) * 0.2 + 
            collaboration * 0.1
        )
        new_task["efficiency"] = output_quality / new_task["days"]
        new_task["velocity"] = complexity / new_task["days"]

        # Append to data
        self.data = pd.concat([self.data, pd.DataFrame([new_task])], ignore_index=True)

        # Re-train models
        self._train_models()

        # Update state
        self.state["total_tasks"] = len(self.data)
        self.state["total_days"] = self.data["days"].sum()
        self.state["last_updated"] = datetime.now().isoformat()

        print(f"✅ Task added: {task_name} ({category})")
        print(f"   Productivity Score: {new_task['productivity_score']:.2f}")
        print(f"   Efficiency: {new_task['efficiency']:.2f}")
        return new_task

    def _train_models(self):
        """Internal: Train all ML models on current data."""
        if len(self.data) < 3:
            print("⚠️  Need at least 3 tasks to train models.")
            return

        df = self.data.copy()
        df['month'] = df['start_date'].dt.month

        # Encode categories
        if 'category' in df.columns:
            self.le.fit(df['category'])
            df['category_encoded'] = self.le.transform(df['category'])

        # Features
        features = ['complexity', 'category_encoded', 'month', 'collaboration']
        X = df[features].fillna(0)
        y = df['productivity_score']

        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Model 1: Productivity Predictor
        self.models['productivity'] = RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=5
        )
        self.models['productivity'].fit(X_scaled, y)

        # Model 2: Peak Detector
        self.models['peak'] = IsolationForest(
            contamination=0.25, random_state=42
        )
        self.models['peak'].fit(df[['productivity_score', 'efficiency', 'velocity']])

        # Model 3: Pattern Clustering
        cluster_features = df[['complexity', 'productivity_score', 'efficiency', 'collaboration']].values
        cluster_scaled = StandardScaler().fit_transform(cluster_features)
        self.models['cluster'] = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.models['cluster'].fit(cluster_scaled)

        # Update predictions on data
        df['is_peak'] = self.models['peak'].predict(df[['productivity_score', 'efficiency', 'velocity']])
        df['is_peak'] = df['is_peak'].map({1: 0, -1: 1})

        clusters = self.models['cluster'].predict(cluster_scaled)
        df['cluster'] = clusters

        # Rank clusters
        cluster_means = df.groupby('cluster')['productivity_score'].mean().sort_values(ascending=False)
        rank_map = {old: new for new, old in enumerate(cluster_means.index)}
        df['cluster_rank'] = df['cluster'].map(rank_map)
        cluster_labels = {0: 'High Achiever', 1: 'Steady Performer', 2: 'Growth Zone'}
        df['cluster_label'] = df['cluster_rank'].map(cluster_labels)

        self.data = df
        self.state["peak_count"] = int(df['is_peak'].sum())

        # Trend analysis
        if len(df) >= 3:
            df['task_num'] = range(1, len(df)+1)
            slope, _, r_value, _, _ = stats.linregress(df['task_num'], df['productivity_score'])
            self.state["trend_slope"] = round(slope, 3)

        print(f"🧠 Models re-trained on {len(df)} tasks.")

    def predict_next_task(self, category, complexity, collaboration, month=None):
        """
        Predict productivity for a hypothetical upcoming task.

        Returns predicted score and confidence level.
        """
        if 'productivity' not in self.models:
            return {"error": "No trained model available. Add more tasks first."}

        if month is None:
            month = datetime.now().month

        # Encode category
        try:
            cat_encoded = self.le.transform([category])[0]
        except ValueError:
            cat_encoded = 0  # Default for unseen categories

        task_features = np.array([[complexity, cat_encoded, month, collaboration]])
        task_scaled = self.scaler.transform(task_features)

        prediction = self.models['productivity'].predict(task_scaled)[0]

        # Determine confidence based on historical similar tasks
        similar = self.data[
            (self.data['category'] == category) & 
            (abs(self.data['complexity'] - complexity) <= 2)
        ]

        confidence = "HIGH" if len(similar) >= 3 else "MEDIUM" if len(similar) >= 1 else "LOW"

        return {
            "predicted_score": round(prediction, 2),
            "confidence": confidence,
            "similar_historical_tasks": len(similar),
            "recommendation": self._get_task_recommendation(prediction, complexity, collaboration)
        }

    def _get_task_recommendation(self, score, complexity, collaboration):
        """Generate contextual recommendation for a task."""
        if score >= 7.5:
            return "🔥 This task aligns with your peak performance profile. Go for it!"
        elif score >= 6.0:
            return "✅ Good fit. Consider increasing collaboration to boost score."
        else:
            return "⚠️  This may be challenging. Break it down or pair with a teammate."

    def get_peak_performance_report(self):
        """Identify when and what you do best."""
        peaks = self.data[self.data['is_peak'] == 1]

        if len(peaks) == 0:
            return {"message": "No peak performance detected yet. Keep adding tasks!"}

        report = {
            "total_peak_tasks": len(peaks),
            "peak_tasks": peaks['task'].tolist(),
            "peak_categories": peaks['category'].unique().tolist(),
            "avg_complexity_during_peak": round(peaks['complexity'].mean(), 1),
            "avg_collaboration_during_peak": round(peaks['collaboration'].mean(), 1),
            "peak_dates": peaks['start_date'].dt.strftime('%Y-%m-%d').tolist(),
            "pattern": ""
        }

        # Detect pattern
        if peaks['complexity'].mean() > 8:
            report["pattern"] = "You peak on HIGH-complexity challenges."
        elif peaks['collaboration'].mean() > 7:
            report["pattern"] = "You peak when collaborating with others."
        else:
            report["pattern"] = "You peak on balanced, well-supported tasks."

        return report

    def get_skill_profile(self):
        """Generate your skill strength profile."""
        if len(self.data) == 0:
            return {}

        profile = self.data.groupby('category').agg({
            'productivity_score': 'mean',
            'efficiency': 'mean',
            'output_quality': 'mean',
            'satisfaction': 'mean',
            'task': 'count'
        }).round(2)

        profile['composite'] = (
            profile['productivity_score'] * 0.4 + 
            profile['efficiency'] * 10 * 0.2 + 
            profile['output_quality'] * 0.2 + 
            profile['satisfaction'] * 0.2
        ).round(2)

        profile = profile.sort_values('composite', ascending=False)

        return {
            "strengths": profile.head(2).to_dict(),
            "weaknesses": profile.tail(1).to_dict(),
            "ranking": profile['composite'].to_dict(),
            "total_categories": len(profile)
        }

    def detect_burnout_risk(self):
        """
        Analyze recent tasks for burnout indicators:
        - Declining satisfaction despite maintained output
        - Efficiency drops on familiar tasks
        - Increasing days per task
        """
        if len(self.data) < 4:
            return {"risk": "UNKNOWN", "reason": "Not enough data"}

        recent = self.data.tail(4)
        older = self.data.head(len(self.data) - 4)

        risks = []

        # Check efficiency drop
        if recent['efficiency'].mean() < older['efficiency'].mean() * 0.8:
            risks.append("Efficiency drop detected")

        # Check satisfaction decline
        if len(recent) >= 3:
            sat_slope, _, _, _, _ = stats.linregress(range(len(recent)), recent['satisfaction'])
            if sat_slope < -0.5:
                risks.append("Satisfaction declining")

        # Check task duration increase
        if recent['days'].mean() > older['days'].mean() * 1.5:
            risks.append("Tasks taking longer than usual")

        if len(risks) >= 2:
            self.state["burnout_risk"] = "HIGH"
            return {"risk": "HIGH", "indicators": risks, "advice": "Take a break. Reduce complexity for next 2 tasks."}
        elif len(risks) == 1:
            self.state["burnout_risk"] = "MEDIUM"
            return {"risk": "MEDIUM", "indicators": risks, "advice": "Monitor closely. Add variety to your work."}
        else:
            self.state["burnout_risk"] = "LOW"
            return {"risk": "LOW", "indicators": [], "advice": "You're in a good zone. Keep going!"}

    def get_optimal_schedule(self, upcoming_tasks):
        """
        Recommend optimal sequencing for upcoming tasks.

        upcoming_tasks: list of dicts with 'task', 'category', 'complexity', 'collaboration'
        """
        predictions = []
        month = datetime.now().month

        for task in upcoming_tasks:
            pred = self.predict_next_task(
                task['category'], 
                task['complexity'], 
                task['collaboration'], 
                month
            )
            predictions.append({
                **task,
                **pred
            })

        # Sort by predicted score descending
        predictions.sort(key=lambda x: x['predicted_score'], reverse=True)

        # Add sequencing advice
        for i, task in enumerate(predictions):
            if i == 0:
                task['schedule_advice'] = "🥇 Start here — highest predicted performance"
            elif task['predicted_score'] < 6.0:
                task['schedule_advice'] = f"⏳ Schedule later or pair with support"
            else:
                task['schedule_advice'] = f"✅ Good mid-sequence task"

        return predictions

    def learn_from_web(self, search_results):
        """
        Integrate external knowledge into recommendations.
        search_results: list of dicts with 'title', 'snippet'
        """
        new_knowledge = []
        for result in search_results:
            knowledge = {
                'source': result.get('title', 'Unknown'),
                'insight': result.get('snippet', '')[:200],
                'timestamp': datetime.now().isoformat(),
                'integrated': False
            }
            self.knowledge_base.append(knowledge)
            new_knowledge.append(knowledge)

        print(f"🌐 Learned {len(new_knowledge)} new insights from web.")
        return new_knowledge

    def generate_full_report(self):
        """Generate comprehensive productivity report."""
        report = {
            "agent_version": self.version,
            "user": self.user_name,
            "generated_at": datetime.now().isoformat(),
            "state": self.state,
            "peak_performance": self.get_peak_performance_report(),
            "skill_profile": self.get_skill_profile(),
            "burnout_status": self.detect_burnout_risk(),
            "trend": "improving" if self.state["trend_slope"] > 0.05 else "stable" if self.state["trend_slope"] > -0.05 else "declining",
            "recommendations": self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self):
        """Internal: Generate personalized recommendations."""
        recs = []

        # Skill-based
        profile = self.get_skill_profile()
        if profile.get('ranking'):
            top_skill = max(profile['ranking'], key=profile['ranking'].get)
            bottom_skill = min(profile['ranking'], key=profile['ranking'].get)
            recs.append(f"💪 Double down on {top_skill} — your strongest area.")
            recs.append(f"📈 Invest in {bottom_skill} — biggest growth opportunity.")

        # Collaboration
        if 'collaboration' in self.data.columns:
            corr = self.data['collaboration'].corr(self.data['productivity_score'])
            if corr and corr > 0.5:
                recs.append(f"🔗 Collaboration strongly boosts your output (r={corr:.2f}). Seek team projects.")

        # Trend
        if self.state["trend_slope"] > 0.1:
            recs.append("📈 You're in a growth phase. Take on bigger challenges.")
        elif self.state["trend_slope"] < -0.1:
            recs.append("⚠️  Productivity trending down. Mix up your task types.")

        # Burnout
        if self.state["burnout_risk"] == "HIGH":
            recs.append("🛑 BURNOUT RISK: Take 3-5 days lighter workload immediately.")

        # Web knowledge integration
        for knowledge in self.knowledge_base[-3:]:
            if not knowledge['integrated']:
                recs.append(f"🌐 WEB-INSIGHT: {knowledge['insight'][:100]}...")
                knowledge['integrated'] = True

        return recs

    def export_data(self, filepath="productivity_data.json"):
        """Export all data and state to JSON."""
        export = {
            "user": self.user_name,
            "version": self.version,
            "state": self.state,
            "tasks": self.data.to_dict('records') if len(self.data) > 0 else [],
            "knowledge_base": self.knowledge_base,
            "insights_history": self.insights_history
        }

        # Convert timestamps to strings for JSON
        for task in export["tasks"]:
            if isinstance(task.get('start_date'), pd.Timestamp):
                task['start_date'] = task['start_date'].strftime('%Y-%m-%d')
            if isinstance(task.get('end_date'), pd.Timestamp):
                task['end_date'] = task['end_date'].strftime('%Y-%m-%d')

        with open(filepath, 'w') as f:
            json.dump(export, f, indent=2, default=str)

        print(f"💾 Data exported to {filepath}")
        return filepath

    def import_data(self, filepath="productivity_data.json"):
        """Import previously saved data."""
        if not os.path.exists(filepath):
            print(f"⚠️  File {filepath} not found.")
            return False

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.user_name = data.get('user', self.user_name)
        self.state = data.get('state', self.state)
        self.knowledge_base = data.get('knowledge_base', [])
        self.insights_history = data.get('insights_history', [])

        if data.get('tasks'):
            self.data = pd.DataFrame(data['tasks'])
            self.data['start_date'] = pd.to_datetime(self.data['start_date'])
            self.data['end_date'] = pd.to_datetime(self.data['end_date'])
            self._train_models()

        print(f"📂 Imported {len(self.data)} tasks from {filepath}")
        return True


# ============================================
# DEMONSTRATION / USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 PRODUCTIVITY INTELLIGENCE AGENT v2.0")
    print("   Your Personal AI Work Coach")
    print("=" * 60)

    # Initialize agent
    agent = ProductivityIntelligenceAgent(user_name="Structural Engineer")

    # Add historical tasks (your 3 weeks examples + more)
    tasks = [
        ("Design soffits and connections for balconies", "Structural Design", "2026-01-06", "2026-01-27", 8, 7, 8, 5),
        ("Design cassettes and prepare drawings", "Structural Design", "2026-01-28", "2026-02-18", 7, 6, 7, 4),
        ("Foundation analysis for high-rise", "Analysis", "2026-02-19", "2026-03-05", 9, 9, 9, 6),
        ("Beam reinforcement detailing", "Detailing", "2026-03-06", "2026-03-12", 5, 5, 6, 3),
        ("Load calculations for wind analysis", "Analysis", "2026-03-13", "2026-03-26", 8, 8, 8, 7),
        ("Steel connection design", "Structural Design", "2026-03-27", "2026-04-09", 9, 9, 9, 8),
        ("Concrete mix design review", "Materials", "2026-04-10", "2026-04-16", 6, 6, 7, 4),
        ("Seismic retrofit assessment", "Analysis", "2026-04-17", "2026-05-08", 10, 10, 10, 9),
    ]

    for t in tasks:
        agent.add_task(*t)

    print("\n" + "=" * 60)
    print("📊 FULL PRODUCTIVITY REPORT")
    print("=" * 60)

    report = agent.generate_full_report()

    print(f"\n👤 User: {report['user']}")
    print(f"📅 Report Generated: {report['generated_at'][:19]}")
    print(f"📈 Trend: {report['trend'].upper()}")
    print(f"🔥 Peak Tasks: {report['state']['peak_count']}")
    print(f"🎯 Total Tasks: {report['state']['total_tasks']}")
    print(f"📅 Total Days: {report['state']['total_days']}")

    print("\n🔥 PEAK PERFORMANCE:")
    peak = report['peak_performance']
    if 'tasks' in peak:
        print(f"   Tasks: {', '.join(peak['tasks'])}")
        print(f"   Pattern: {peak['pattern']}")

    print("\n💪 SKILL PROFILE:")
    profile = report['skill_profile']
    if 'ranking' in profile:
        for skill, score in profile['ranking'].items():
            bar = "█" * int(score / 2)
            print(f"   {skill:20s} {score:.1f} {bar}")

    print("\n🩺 BURNOUT STATUS:")
    burnout = report['burnout_status']
    print(f"   Risk: {burnout['risk']}")
    print(f"   Advice: {burnout['advice']}")

    print("\n🎯 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"   {rec}")

    # Predict next tasks
    print("\n" + "=" * 60)
    print("🔮 PREDICTIONS FOR UPCOMING TASKS")
    print("=" * 60)

    upcoming = [
        {"task": "Bridge deck analysis", "category": "Analysis", "complexity": 8, "collaboration": 7},
        {"task": "Column detailing for warehouse", "category": "Detailing", "complexity": 6, "collaboration": 5},
        {"task": "Tower foundation design", "category": "Structural Design", "complexity": 9, "collaboration": 8},
    ]

    schedule = agent.get_optimal_schedule(upcoming)

    for i, task in enumerate(schedule, 1):
        print(f"\n   {i}. {task['task']} ({task['category']})")
        print(f"      Predicted Score: {task['predicted_score']:.2f} (Confidence: {task['confidence']})")
        print(f"      {task['recommendation']}")
        print(f"      {task['schedule_advice']}")

    # Export
    print("\n" + "=" * 60)
    agent.export_data("my_productivity_data.json")
    print("\n✅ Agent ready! Add new tasks anytime with agent.add_task(...)")
