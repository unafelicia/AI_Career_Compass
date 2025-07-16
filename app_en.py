import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🧭 AI Career Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        overflow: hidden;
    }
    .metric-card h2 {
        font-size: 1.5rem;
        margin: 0.5rem 0;
        word-wrap: break-word;
    }
    .metric-card h3 {
        font-size: 1rem;
        margin: 0.3rem 0;
    }
    .metric-card p {
        font-size: 0.9rem;
        margin: 0.2rem 0;
    }
    .metric-card small {
        font-size: 0.8rem;
    }
    .advice-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #2E86AB;
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    .advice-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: bold;
        color: #2E86AB;
    }
    .advice-content {
        color: #333;
        line-height: 1.6;
    }
    .skill-compact-row {
        background: #f8f9fa;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border: 1px solid #e0e0e0;
    }
    .skill-resource-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .skill-resource-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    .resource-link {
        color: #2E86AB;
        text-decoration: none;
        font-weight: 500;
    }
    .resource-link:hover {
        color: #A23B72;
    }
    .author-footer {
        background: #f8f9fa;
        padding: 2rem;
        border-top: 1px solid #e0e0e0;
        text-align: center;
        margin-top: 3rem;
        color: #666;
    }
    .author-footer a {
        color: #2E86AB;
        text-decoration: none;
    }
    .author-footer a:hover {
        color: #A23B72;
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}
if 'skills_list' not in st.session_state:
    st.session_state.skills_list = [{"skill": "", "proficiency": 50} for _ in range(3)]

# Predefined skills list
PREDEFINED_SKILLS = [
    "Python", "SQL", "Machine Learning", "Deep Learning", "Data Analysis",
    "Statistics", "Pandas", "NumPy", "TensorFlow", "PyTorch", "Scikit-learn",
    "Tableau", "Power BI", "Excel", "R", "Java", "Spark", "Hadoop", "AWS",
    "Azure", "GCP", "Docker", "Kubernetes", "Git", "Linux", "NoSQL", "MongoDB",
    "PostgreSQL", "MySQL", "ETL", "Data Visualization", "Business Intelligence",
    "Natural Language Processing", "Computer Vision", "Time Series Analysis",
    "A/B Testing", "Statistical Modeling", "Neural Networks", "API Development",
    "Web Scraping", "Data Mining", "Big Data", "Cloud Computing", "MLOps"
]

# Tiered learning resources
LEARNING_RESOURCES = {
    "Python": {
        "beginner": [
            {"name": "Python Official Tutorial", "url": "https://docs.python.org/3/tutorial/", "type": "Documentation"},
            {"name": "Automate the Boring Stuff", "url": "https://automatetheboringstuff.com/", "type": "Online Book"},
            {"name": "Python for Everybody (Coursera)", "url": "https://www.coursera.org/specializations/python",
             "type": "Course"}
        ],
        "intermediate": [
            {"name": "Real Python Advanced Tutorials", "url": "https://realpython.com/tutorials/advanced/",
             "type": "Tutorial"},
            {"name": "Python Tricks Book", "url": "https://realpython.com/python-tricks/", "type": "Advanced Book"},
            {"name": "Advanced Python Programming",
             "url": "https://www.udemy.com/course/python-beyond-the-basics-object-oriented-programming/",
             "type": "Course"}
        ],
        "advanced": [
            {"name": "Python Internals", "url": "https://github.com/python/cpython/tree/main/Doc",
             "type": "Source Study"},
            {"name": "High Performance Python",
             "url": "https://www.oreilly.com/library/view/high-performance-python/9781492055013/",
             "type": "Expert Book"},
            {"name": "Python C API", "url": "https://docs.python.org/3/extending/", "type": "Extension Dev"}
        ]
    },
    "Machine Learning": {
        "beginner": [
            {"name": "Andrew Ng ML Course", "url": "https://www.coursera.org/learn/machine-learning", "type": "Course"},
            {"name": "Scikit-learn User Guide", "url": "https://scikit-learn.org/stable/user_guide.html",
             "type": "Documentation"},
            {"name": "Kaggle Learn ML", "url": "https://www.kaggle.com/learn/intro-to-machine-learning",
             "type": "Hands-on Tutorial"}
        ],
        "intermediate": [
            {"name": "Hands-On ML with Scikit-Learn", "url": "https://github.com/ageron/handson-ml2",
             "type": "Practical Project"},
            {"name": "Feature Engineering Course", "url": "https://www.coursera.org/learn/feature-engineering",
             "type": "Professional Course"},
            {"name": "ML Engineering for Production",
             "url": "https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops",
             "type": "Engineering Practice"}
        ],
        "advanced": [
            {"name": "Advanced ML Specialization", "url": "https://www.coursera.org/specializations/aml",
             "type": "Advanced Specialization"},
            {"name": "ML Research Papers", "url": "https://paperswithcode.com/", "type": "Cutting-edge Research"},
            {"name": "Custom ML Algorithms", "url": "https://github.com/rushter/MLAlgorithms",
             "type": "Algorithm Implementation"}
        ]
    },
    "Deep Learning": {
        "beginner": [
            {"name": "Deep Learning Specialization", "url": "https://www.coursera.org/specializations/deep-learning",
             "type": "Course"},
            {"name": "PyTorch Tutorials", "url": "https://pytorch.org/tutorials/", "type": "Documentation"},
            {"name": "Fast.ai Practical DL", "url": "https://course.fast.ai/", "type": "Practical Course"}
        ],
        "intermediate": [
            {"name": "Advanced PyTorch", "url": "https://pytorch.org/tutorials/intermediate/",
             "type": "Advanced Tutorial"},
            {"name": "CS231n Stanford", "url": "http://cs231n.stanford.edu/", "type": "Academic Course"},
            {"name": "Deep Learning Book", "url": "https://www.deeplearningbook.org/", "type": "Theoretical Foundation"}
        ],
        "advanced": [
            {"name": "Transformer Architecture",
             "url": "https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf",
             "type": "Research Paper"},
            {"name": "Advanced GAN Techniques", "url": "https://github.com/eriklindernoren/PyTorch-GAN",
             "type": "Cutting-edge Implementation"},
            {"name": "Neural Architecture Search", "url": "https://arxiv.org/abs/1808.05377",
             "type": "Research Frontier"}
        ]
    },
    "SQL": {
        "beginner": [
            {"name": "SQLBolt Interactive", "url": "https://sqlbolt.com/", "type": "Interactive Tutorial"},
            {"name": "W3Schools SQL", "url": "https://www.w3schools.com/sql/", "type": "Basic Tutorial"},
            {"name": "SQL for Data Science", "url": "https://www.coursera.org/learn/sql-for-data-science",
             "type": "Course"}
        ],
        "intermediate": [
            {"name": "Advanced SQL Techniques", "url": "https://mode.com/sql-tutorial/advanced/",
             "type": "Advanced Tutorial"},
            {"name": "SQL Performance Tuning", "url": "https://use-the-index-luke.com/",
             "type": "Performance Optimization"},
            {"name": "Window Functions Deep Dive",
             "url": "https://www.postgresql.org/docs/current/tutorial-window.html", "type": "Specialized Tutorial"}
        ],
        "advanced": [
            {"name": "Query Optimization", "url": "https://www.postgresql.org/docs/current/planner-optimizer.html",
             "type": "Optimizer Principles"},
            {"name": "Database Internals", "url": "https://www.databass.dev/", "type": "Database Kernel"},
            {"name": "Distributed SQL Systems",
             "url": "https://architecture-center.github.io/azure-architecture-center/data-guide/relational-data/",
             "type": "Distributed Architecture"}
        ]
    },
    "Data Analysis": {
        "beginner": [
            {"name": "Pandas Getting Started", "url": "https://pandas.pydata.org/docs/getting_started/index.html",
             "type": "Documentation"},
            {"name": "Data Analysis with Python", "url": "https://www.coursera.org/learn/data-analysis-with-python",
             "type": "Course"},
            {"name": "Kaggle Data Cleaning", "url": "https://www.kaggle.com/learn/data-cleaning", "type": "Hands-on"}
        ],
        "intermediate": [
            {"name": "Advanced Pandas", "url": "https://pandas.pydata.org/docs/user_guide/advanced.html",
             "type": "Advanced Documentation"},
            {"name": "Statistical Data Analysis",
             "url": "https://www.coursera.org/specializations/statistics-with-python", "type": "Statistical Analysis"},
            {"name": "Time Series Analysis", "url": "https://www.kaggle.com/learn/time-series",
             "type": "Specialized Analysis"}
        ],
        "advanced": [
            {"name": "Big Data Analytics", "url": "https://spark.apache.org/docs/latest/sql-programming-guide.html",
             "type": "Big Data Analysis"},
            {"name": "Advanced Statistics",
             "url": "https://online.stanford.edu/courses/stats200-introduction-statistical-inference",
             "type": "Advanced Statistics"},
            {"name": "Causal Inference", "url": "https://mixtape.scunning.com/", "type": "Causal Inference"}
        ]
    },
    "Cloud Computing": {
        "beginner": [
            {"name": "AWS Getting Started", "url": "https://aws.amazon.com/getting-started/",
             "type": "Official Tutorial"},
            {"name": "Cloud Fundamentals", "url": "https://www.coursera.org/learn/introduction-to-cloud",
             "type": "Foundation Course"},
            {"name": "Azure Fundamentals", "url": "https://docs.microsoft.com/en-us/learn/paths/azure-fundamentals/",
             "type": "Microsoft Certification"}
        ],
        "intermediate": [
            {"name": "AWS Solutions Architect",
             "url": "https://aws.amazon.com/certification/certified-solutions-architect-associate/",
             "type": "Professional Certification"},
            {"name": "Kubernetes Deep Dive", "url": "https://kubernetes.io/docs/tutorials/",
             "type": "Container Orchestration"},
            {"name": "DevOps with Cloud",
             "url": "https://www.coursera.org/specializations/devops-cloud-and-agile-foundations",
             "type": "DevOps Practice"}
        ],
        "advanced": [
            {"name": "Cloud Architecture Patterns", "url": "https://docs.microsoft.com/en-us/azure/architecture/",
             "type": "Architecture Design"},
            {"name": "Multi-Cloud Strategy", "url": "https://cloud.google.com/architecture/framework",
             "type": "Multi-Cloud Strategy"},
            {"name": "Serverless Computing", "url": "https://martinfowler.com/articles/serverless.html",
             "type": "Serverless"}
        ]
    },
    "MLOps": {
        "beginner": [
            {"name": "MLOps Fundamentals", "url": "https://ml-ops.org/content/motivation",
             "type": "Concept Introduction"},
            {"name": "MLflow Tutorial", "url": "https://mlflow.org/docs/latest/tutorials-and-examples/index.html",
             "type": "Tool Tutorial"},
            {"name": "ML Pipelines Intro",
             "url": "https://www.coursera.org/learn/machine-learning-engineering-for-production-mlops",
             "type": "Pipeline"}
        ],
        "intermediate": [
            {"name": "Kubeflow Pipelines", "url": "https://www.kubeflow.org/docs/components/pipelines/",
             "type": "Production Deployment"},
            {"name": "Model Monitoring",
             "url": "https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/",
             "type": "Model Monitoring"},
            {"name": "A/B Testing for ML",
             "url": "https://booking.ai/how-booking-com-increases-the-power-of-online-experiments-with-machine-learning-25aa102df618",
             "type": "Experiment Design"}
        ],
        "advanced": [
            {"name": "ML System Design", "url": "https://github.com/chiphuyen/machine-learning-systems-design",
             "type": "System Design"},
            {"name": "Feature Stores", "url": "https://www.tecton.ai/blog/what-is-a-feature-store/",
             "type": "Feature Engineering"},
            {"name": "Real-time ML", "url": "https://huyenchip.com/2020/12/27/real-time-machine-learning.html",
             "type": "Real-time Systems"}
        ]
    }
}

# School tier weights
SCHOOL_WEIGHTS = {
    "Top Tier (Ivy League/QS Top 50)": 1.2,
    "Second Tier (QS 51-200)": 1.05,
    "Other": 1.0
}

# Industry weights
INDUSTRY_WEIGHTS = {
    "Technology": 1.1,
    "Finance": 1.2,
    "Healthcare": 1.0,
    "Media": 0.9,
    "Retail": 0.8,
    "Energy": 0.95,
    "Other": 1.0
}


# Load models
@st.cache_resource
def load_models():
    try:
        salary_model = joblib.load('salary_prediction_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        return salary_model, label_encoders
    except FileNotFoundError:
        st.error("Model files not found. Please ensure model files are in the correct path")
        return None, None


# Enhanced skill clustering function
def analyze_skills_cluster(skills_data):
    """Cluster analysis based on skill combinations and proficiency"""
    if not skills_data or not any(skill['skill'] for skill in skills_data):
        return "Generalist", []

    # Define career type skill patterns
    skill_patterns = {
        "Data Scientist": {
            "core_skills": ["python", "machine learning", "statistics", "data analysis"],
            "bonus_skills": ["pandas", "numpy", "scikit-learn", "jupyter"],
            "min_core_count": 2,
            "weight_factor": 1.0
        },
        "AI Engineer": {
            "core_skills": ["deep learning", "tensorflow", "pytorch", "neural networks"],
            "bonus_skills": ["computer vision", "nlp", "python", "machine learning"],
            "min_core_count": 2,
            "weight_factor": 1.1
        },
        "Data Engineer": {
            "core_skills": ["sql", "spark", "hadoop", "etl"],
            "bonus_skills": ["aws", "azure", "docker", "kubernetes", "python"],
            "min_core_count": 2,
            "weight_factor": 1.05
        },
        "Business Analyst": {
            "core_skills": ["excel", "tableau", "power bi", "business intelligence"],
            "bonus_skills": ["sql", "analytics", "data visualization"],
            "min_core_count": 1,
            "weight_factor": 0.95
        }
    }

    user_skills = []
    total_proficiency = 0

    # Extract user skills and proficiency
    for skill_item in skills_data:
        if skill_item['skill']:
            skill_lower = skill_item['skill'].lower()
            proficiency = skill_item['proficiency']
            user_skills.append((skill_lower, proficiency))
            total_proficiency += proficiency

    if not user_skills:
        return "Generalist", PREDEFINED_SKILLS[:5]

    # Calculate match score for each career type
    cluster_scores = {}

    for cluster_name, pattern in skill_patterns.items():
        core_matches = 0
        core_proficiency_sum = 0
        bonus_score = 0

        # Check core skill matches
        for skill_lower, proficiency in user_skills:
            for core_skill in pattern["core_skills"]:
                if core_skill in skill_lower:
                    core_matches += 1
                    core_proficiency_sum += proficiency
                    break

            # Check bonus skills
            for bonus_skill in pattern["bonus_skills"]:
                if bonus_skill in skill_lower:
                    bonus_score += proficiency * 0.5
                    break

        # Only consider career types that meet minimum core skill requirements
        if core_matches >= pattern["min_core_count"]:
            # Calculate comprehensive score
            core_score = (core_proficiency_sum / max(core_matches, 1)) * core_matches
            total_score = (core_score + bonus_score) * pattern["weight_factor"]
            cluster_scores[cluster_name] = total_score
        else:
            cluster_scores[cluster_name] = 0

    # Return generalist if no clear career type match
    if not cluster_scores or max(cluster_scores.values()) == 0:
        return "Generalist", PREDEFINED_SKILLS[:5]

    # Return highest scoring career type
    best_cluster = max(cluster_scores, key=cluster_scores.get)
    recommended_skills = (skill_patterns[best_cluster]["core_skills"] +
                          skill_patterns[best_cluster]["bonus_skills"])[:5]

    return best_cluster, recommended_skills


# Salary prediction function
def predict_salary(user_data, skills_data, salary_model, label_encoders):
    """Predict user salary including school and skill weights"""
    if not salary_model or not label_encoders:
        base_salary = calculate_base_salary(user_data, skills_data)
        return base_salary, base_salary * 0.85, base_salary * 1.15

    try:
        base_salary = calculate_base_salary(user_data, skills_data)

        # Apply school weight
        school_weight = SCHOOL_WEIGHTS.get(user_data.get('school_tier', 'Other'), 1.0)

        # Apply industry weight
        industry_weight = INDUSTRY_WEIGHTS.get(user_data.get('industry', 'Other'), 1.0)

        # Calculate skill weight
        skill_weight = calculate_skill_weight(skills_data)

        # Comprehensive calculation
        predicted_salary = base_salary * school_weight * industry_weight * skill_weight

        # Calculate confidence interval
        lower_bound = predicted_salary * 0.85
        upper_bound = predicted_salary * 1.15

        return predicted_salary, lower_bound, upper_bound

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        base_salary = calculate_base_salary(user_data, skills_data)
        return base_salary, base_salary * 0.85, base_salary * 1.15


def calculate_base_salary(user_data, skills_data):
    """Calculate base salary"""
    base_salaries = {
        "Entry Level (0-2 years)": 70000,
        "Mid Level (2-5 years)": 95000,
        "Senior Level (5-10 years)": 130000,
        "Expert Level (10+ years)": 180000
    }

    experience_level = user_data.get('experience_level', 'Mid Level (2-5 years)')
    base = base_salaries.get(experience_level, 95000)

    # Location adjustment
    location_multipliers = {
        "United States": 1.0, "United Kingdom": 0.8, "Canada": 0.85,
        "Germany": 0.75, "Netherlands": 0.8, "China": 0.6, "Singapore": 0.9
    }

    location = user_data.get('company_location', 'United States')
    base *= location_multipliers.get(location, 0.8)

    # Company size adjustment
    size_multipliers = {
        "Small Company (<50 employees)": 0.9,
        "Medium Company (50-250 employees)": 1.0,
        "Large Company (>250 employees)": 1.15
    }

    company_size = user_data.get('company_size', 'Medium Company (50-250 employees)')
    base *= size_multipliers.get(company_size, 1.0)

    # Remote work adjustment
    remote_multipliers = {
        "0% (Fully On-site)": 1.0, "25% (Occasional Remote)": 1.02,
        "50% (Hybrid)": 1.05, "75% (Mostly Remote)": 1.08, "100% (Fully Remote)": 1.1
    }

    remote_ratio = user_data.get('remote_ratio', '50% (Hybrid)')
    base *= remote_multipliers.get(remote_ratio, 1.0)

    return int(base)


def calculate_skill_weight(skills_data):
    """Calculate weight based on skill count and proficiency"""
    if not skills_data:
        return 1.0

    total_score = 0
    skill_count = 0

    for skill_item in skills_data:
        if skill_item['skill']:
            proficiency = skill_item['proficiency']
            total_score += proficiency
            skill_count += 1

    if skill_count == 0:
        return 1.0

    avg_proficiency = total_score / skill_count
    skill_weight = 0.8 + (avg_proficiency / 100) * 0.4
    skill_count_bonus = min(skill_count * 0.02, 0.1)

    return skill_weight + skill_count_bonus


# Generate career advice
def generate_career_advice(user_profile, cluster_type, salary_range, skills_data):
    """Generate personalized career advice"""
    advice_templates = {
        "Data Scientist": {
            "strengths": "You have a solid foundation in data analysis and machine learning",
            "gaps": "Consider strengthening deep learning and big data processing skills",
            "next_steps": "Consider learning TensorFlow/PyTorch and getting AWS certification",
            "icon": "📊"
        },
        "AI Engineer": {
            "strengths": "You have strong technical capabilities in AI technology stack",
            "gaps": "Consider strengthening productization and engineering practice experience",
            "next_steps": "Participate in open source projects, learn MLOps and model deployment",
            "icon": "🤖"
        },
        "Data Engineer": {
            "strengths": "You have good skills in data infrastructure",
            "gaps": "Consider learning more cloud platforms and real-time data processing technologies",
            "next_steps": "Dive deep into Kafka, Kubernetes and other technologies",
            "icon": "🔧"
        },
        "Business Analyst": {
            "strengths": "You have advantages in business understanding and data visualization",
            "gaps": "Consider strengthening programming skills and statistical analysis capabilities",
            "next_steps": "Learn Python/R and master advanced analysis methods",
            "icon": "📈"
        },
        "Generalist": {
            "strengths": "You have a diverse skill foundation across multiple areas",
            "gaps": "Consider focusing on a specific domain for deeper development",
            "next_steps": "Choose an area of interest and systematically improve professional skills",
            "icon": "🎯"
        }
    }

    template = advice_templates.get(cluster_type, advice_templates["Generalist"])

    return template


# Compact skill input component
def skill_input_component():
    """Compact skill input component"""
    with st.expander("🎯 Skills Configuration", expanded=True):
        st.write("Select your skills and evaluate proficiency levels:")

        # Add skill button
        if st.button("➕ Add Skill", key="add_skill_btn"):
            st.session_state.skills_list.append({"skill": "", "proficiency": 50})
            st.rerun()

        # Skill input rows (compact layout)
        skills_to_remove = []
        for i, skill_item in enumerate(st.session_state.skills_list):
            with st.container():
                st.markdown(f'<div class="skill-compact-row">', unsafe_allow_html=True)

                col1, col2, col3 = st.columns([4, 3, 1])

                with col1:
                    skill_options = [""] + PREDEFINED_SKILLS + ["Custom..."]
                    current_skill = skill_item.get('skill', '')

                    if current_skill and current_skill not in PREDEFINED_SKILLS:
                        skill_options.insert(-1, current_skill)

                    selected_skill = st.selectbox(
                        f"Skill {i + 1}",
                        options=skill_options,
                        index=skill_options.index(current_skill) if current_skill in skill_options else 0,
                        key=f"skill_{i}",
                        label_visibility="collapsed"
                    )

                    if selected_skill == "Custom...":
                        custom_skill = st.text_input(
                            "",
                            value=current_skill if current_skill not in PREDEFINED_SKILLS else "",
                            key=f"custom_skill_{i}",
                            placeholder="Enter custom skill"
                        )
                        st.session_state.skills_list[i]['skill'] = custom_skill
                    else:
                        st.session_state.skills_list[i]['skill'] = selected_skill

                with col2:
                    proficiency = st.slider(
                        "Proficiency",
                        min_value=0,
                        max_value=100,
                        value=skill_item.get('proficiency', 50),
                        step=5,
                        key=f"proficiency_{i}",
                        label_visibility="collapsed"
                    )
                    st.session_state.skills_list[i]['proficiency'] = proficiency

                with col3:
                    if len(st.session_state.skills_list) > 1:
                        if st.button("🗑️", key=f"remove_{i}", help="Remove skill"):
                            skills_to_remove.append(i)

                st.markdown('</div>', unsafe_allow_html=True)

        # Remove marked skills
        for i in reversed(skills_to_remove):
            st.session_state.skills_list.pop(i)
            st.rerun()


# Display learning resources (with tiered support)
def display_learning_resources(skills, user_skills_data):
    """Display tiered learning resource cards"""
    st.markdown('<div class="sub-header">🎓 Recommended Learning Skills & Resources</div>', unsafe_allow_html=True)

    # Create user skill proficiency dictionary
    user_proficiency = {}
    for skill_item in user_skills_data:
        if skill_item['skill']:
            user_proficiency[skill_item['skill']] = skill_item['proficiency']

    # Ensure at least 4 skills' learning resources are displayed
    display_skills = []

    # Prioritize recommended skills with resources
    for skill in skills:
        if skill.title() in LEARNING_RESOURCES:
            display_skills.append(skill.title())
        elif skill.lower() in [k.lower() for k in LEARNING_RESOURCES.keys()]:
            for resource_key in LEARNING_RESOURCES.keys():
                if skill.lower() == resource_key.lower():
                    display_skills.append(resource_key)
                    break

    # If recommended skills don't have enough resources, supplement with popular skills
    popular_skills = ["Python", "Machine Learning", "SQL", "Data Analysis", "Deep Learning", "Cloud Computing"]
    for skill in popular_skills:
        if len(display_skills) >= 4:
            break
        if skill not in display_skills:
            display_skills.append(skill)

    # Ensure at least 4 skills are displayed
    if len(display_skills) < 4:
        display_skills = ["Python", "Machine Learning", "SQL", "Data Analysis"]

    # Display learning resources for the first 4 skills
    for skill in display_skills[:4]:
        if skill in LEARNING_RESOURCES:
            # Determine resource level based on user proficiency
            proficiency = user_proficiency.get(skill, 0)

            if proficiency >= 80:
                level = "advanced"
                level_name = "Advanced"
                level_color = "#e74c3c"
            elif proficiency >= 60:
                level = "intermediate"
                level_name = "Intermediate"
                level_color = "#f39c12"
            else:
                level = "beginner"
                level_name = "Beginner"
                level_color = "#27ae60"

            st.markdown(
                f"### 📚 {skill} <span style='color: {level_color}; font-size: 0.8em;'>({level_name} Level Resources)</span>",
                unsafe_allow_html=True)

            # Display resources for corresponding level
            if level in LEARNING_RESOURCES[skill]:
                resources = LEARNING_RESOURCES[skill][level]
                cols = st.columns(len(resources))

                for i, resource in enumerate(resources):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="skill-resource-card">
                            <a href="{resource['url']}" target="_blank" class="resource-link">
                                <strong>{resource['name']}</strong>
                            </a>
                            <br><small>{resource['type']}</small>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            # If no specific resources, display general learning advice
            st.markdown(f"### 📚 {skill}")
            st.markdown(f"""
            <div class="skill-resource-card">
                <strong>Recommended Learning for {skill}</strong><br>
                <small>Suggest learning through online courses, official documentation, and hands-on projects</small>
            </div>
            """, unsafe_allow_html=True)


# Main interface
def main():
    st.markdown('<div class="main-header">🧭 AI Career Compass</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 3rem;">AI Job Assistant: Skill Profiling, Salary Prediction, Growth Recommendations</div>',
        unsafe_allow_html=True)

    # Load models
    salary_model, label_encoders = load_models()

    # Sidebar: User input
    with st.sidebar:
        st.header("📝 Personal Information")

        # Basic information
        st.subheader("Background")
        years_experience = st.slider("Years of Experience", 0, 15, 3)

        education_level = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
        school_tier = st.selectbox("School Tier",
                                   ["Top Tier (Ivy League/QS Top 50)", "Second Tier (QS 51-200)", "Other"])

        # Job preferences
        st.subheader("Job Preferences")
        experience_level = st.selectbox("Experience Level", ["Entry Level (0-2 years)", "Mid Level (2-5 years)",
                                                             "Senior Level (5-10 years)", "Expert Level (10+ years)"])
        employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Freelance"])
        company_location = st.selectbox("Company Location",
                                        ["United States", "United Kingdom", "Canada", "Germany", "Netherlands", "China",
                                         "Singapore", "Other"])
        company_size = st.selectbox("Company Size",
                                    ["Small Company (<50 employees)", "Medium Company (50-250 employees)",
                                     "Large Company (>250 employees)"])
        remote_ratio = st.selectbox("Remote Work Ratio",
                                    ["0% (Fully On-site)", "25% (Occasional Remote)", "50% (Hybrid)",
                                     "75% (Mostly Remote)", "100% (Fully Remote)"])
        industry = st.selectbox("Industry",
                                ["Technology", "Finance", "Healthcare", "Media", "Retail", "Energy", "Other"])

        # Analysis button
        if st.button("🚀 Start Analysis", type="primary"):
            st.session_state.user_profile = {
                'years_experience': years_experience,
                'education_required': education_level,
                'school_tier': school_tier,
                'experience_level': experience_level,
                'employment_type': employment_type,
                'company_location': company_location,
                'employee_residence': company_location,
                'company_size': company_size,
                'remote_ratio': remote_ratio,
                'industry': industry
            }
            st.session_state.analysis_complete = True
            st.rerun()

    # Main page content
    if not st.session_state.analysis_complete:
        # Skills input interface (compact version)
        skill_input_component()

        # Welcome information
        st.markdown("""
        ## 🚀 Start Your AI Career Analysis

        Fill in the basic information on the left and skills background above, and we will provide:

        - 🎯 **Intelligent Career Classification** - Professional clustering analysis based on skill combinations
        - 💰 **Accurate Salary Prediction** - Salary model considering school background and skill proficiency
        - 📊 **Visual Skill Profile** - Radar chart showing your skill distribution
        - 🧠 **Personalized Career Advice** - AI-driven development recommendations
        - 🎓 **Learning Resource Recommendations** - Curated skill improvement resource links
        """)

    else:
        # Results display page
        profile = st.session_state.user_profile
        skills_data = st.session_state.skills_list

        # Skill clustering analysis
        cluster_type, recommended_skills = analyze_skills_cluster(skills_data)

        # Salary prediction
        predicted_salary, lower_bound, upper_bound = predict_salary(profile, skills_data, salary_model, label_encoders)

        # Three equal-width metric cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>👤 Career Type</h3>
                <h2>{cluster_type}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Predicted Salary</h3>
                <h2>${predicted_salary:,.0f}</h2>
                <p>${lower_bound:,.0f} - ${upper_bound:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            school_display = profile.get('school_tier', 'Other')
            weight_info = f"Weight: {SCHOOL_WEIGHTS.get(school_display, 1.0):.2f}"
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎓 Education Background</h3>
                <h2>{profile['education_required']}</h2>
                <p>{school_display}</p>
                <small>{weight_info}</small>
            </div>
            """, unsafe_allow_html=True)

        # Detailed analysis
        st.markdown('<div class="sub-header">📊 Detailed Analysis</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Skill Profile")

            # Restore radar chart display
            if skills_data and any(skill['skill'] for skill in skills_data):
                # Create radar chart data
                skills_dict = {}
                for skill_item in skills_data:
                    if skill_item['skill']:
                        skills_dict[skill_item['skill']] = skill_item['proficiency']

                # Supplement common skill dimensions
                radar_skills = ['Programming', 'Machine Learning', 'Data Analysis', 'Statistics', 'Communication',
                                'Domain Knowledge']
                radar_values = []

                for radar_skill in radar_skills:
                    # Calculate scores based on user skill matching
                    score = 0
                    if radar_skill == 'Programming':
                        prog_skills = ['Python', 'R', 'Java', 'SQL']
                        score = max([skills_dict.get(skill, 0) for skill in prog_skills] + [0])
                    elif radar_skill == 'Machine Learning':
                        ml_skills = ['Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch']
                        score = max([skills_dict.get(skill, 0) for skill in ml_skills] + [0])
                    elif radar_skill == 'Data Analysis':
                        da_skills = ['Data Analysis', 'Pandas', 'NumPy', 'Excel']
                        score = max([skills_dict.get(skill, 0) for skill in da_skills] + [0])
                    elif radar_skill == 'Statistics':
                        stat_skills = ['Statistics', 'Statistical Modeling', 'A/B Testing']
                        score = max([skills_dict.get(skill, 0) for skill in stat_skills] + [0])
                    elif radar_skill == 'Communication':
                        # Estimate based on years of experience
                        score = min(50 + (profile['years_experience'] * 5), 100)
                    elif radar_skill == 'Domain Knowledge':
                        # Estimate based on years of experience and industry
                        score = min(40 + (profile['years_experience'] * 4), 100)

                    radar_values.append(score)

                # Create radar chart
                fig = go.Figure()

                fig.add_trace(go.Scatterpolar(
                    r=radar_values,
                    theta=radar_skills,
                    fill='toself',
                    name='Your Skills',
                    line_color='#2E86AB'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=True,
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please add skill information to view skill profile")

        with col2:
            st.subheader("💼 Market Comparison")

            # Salary comparison chart
            comparison_data = {
                'Entry Level': 65000,
                'Mid Level': 85000,
                'Senior Level': 120000,
                'Expert Level': 160000,
                'Your Prediction': predicted_salary
            }

            colors = ['lightblue'] * 4 + ['darkblue']

            fig = px.bar(
                x=list(comparison_data.keys()),
                y=list(comparison_data.values()),
                title="Salary Level Comparison",
                color=colors,
                color_discrete_map="identity"
            )

            fig.update_layout(
                xaxis_title="Career Level",
                yaxis_title="Salary (USD)",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        # AI Career Advice - Card layout
        st.markdown('<div class="sub-header">🧠 AI Career Advice</div>', unsafe_allow_html=True)

        advice_data = generate_career_advice(profile, cluster_type, (predicted_salary, lower_bound, upper_bound),
                                             skills_data)

        # Advice cards
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="advice-card">
                <div class="advice-header">
                    {advice_data['icon']} Core Strengths
                </div>
                <div class="advice-content">
                    {advice_data['strengths']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="advice-card">
                <div class="advice-header">
                    🎯 Development Plan
                </div>
                <div class="advice-content">
                    {advice_data['next_steps']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="advice-card">
                <div class="advice-header">
                    📈 Skill Enhancement
                </div>
                <div class="advice-content">
                    {advice_data['gaps']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Skill analysis
            skill_analysis = ""
            if skills_data and any(skill['skill'] for skill in skills_data):
                high_skills = [s for s in skills_data if s['skill'] and s['proficiency'] >= 80]
                medium_skills = [s for s in skills_data if s['skill'] and 50 <= s['proficiency'] < 80]
                low_skills = [s for s in skills_data if s['skill'] and s['proficiency'] < 50]

                if high_skills:
                    skill_analysis += f"<strong>Expert Skills:</strong> {', '.join([s['skill'] for s in high_skills])}<br>"
                if low_skills:
                    skill_analysis += f"<strong>Need Improvement:</strong> {', '.join([s['skill'] for s in low_skills])}"

            st.markdown(f"""
            <div class="advice-card">
                <div class="advice-header">
                    💰 Salary Insights
                </div>
                <div class="advice-content">
                    Predicted Salary: ${predicted_salary:,.0f} (${lower_bound:,.0f} - ${upper_bound:,.0f})<br>
                    Market Level: {'High' if predicted_salary > 90000 else 'Mid' if predicted_salary > 70000 else 'Entry'} level<br>
                    {skill_analysis}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Learning resource recommendations
        display_learning_resources(recommended_skills, skills_data)

        # Edit skills button
        st.markdown('<div class="sub-header">⚙️ Adjust Settings</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✏️ Edit Skill Configuration", type="secondary"):
                st.session_state.show_skill_editor = True
                st.rerun()

        # Skill editor
        if st.session_state.get('show_skill_editor', False):
            st.markdown("---")
            st.markdown("### ✏️ Skill Configuration Editor")

            # Reuse skill input component
            with st.container():
                st.write("Adjust your skills and re-evaluate proficiency levels:")

                # Add skill button
                if st.button("➕ Add Skill", key="add_skill_btn_edit"):
                    st.session_state.skills_list.append({"skill": "", "proficiency": 50})
                    st.rerun()

                # Skill input rows
                skills_to_remove = []
                for i, skill_item in enumerate(st.session_state.skills_list):
                    with st.container():
                        st.markdown(f'<div class="skill-compact-row">', unsafe_allow_html=True)

                        col1, col2, col3 = st.columns([4, 3, 1])

                        with col1:
                            skill_options = [""] + PREDEFINED_SKILLS + ["Custom..."]
                            current_skill = skill_item.get('skill', '')

                            if current_skill and current_skill not in PREDEFINED_SKILLS:
                                skill_options.insert(-1, current_skill)

                            selected_skill = st.selectbox(
                                f"Skill {i + 1}",
                                options=skill_options,
                                index=skill_options.index(current_skill) if current_skill in skill_options else 0,
                                key=f"skill_edit_{i}",
                                label_visibility="collapsed"
                            )

                            if selected_skill == "Custom...":
                                custom_skill = st.text_input(
                                    "",
                                    value=current_skill if current_skill not in PREDEFINED_SKILLS else "",
                                    key=f"custom_skill_edit_{i}",
                                    placeholder="Enter custom skill"
                                )
                                st.session_state.skills_list[i]['skill'] = custom_skill
                            else:
                                st.session_state.skills_list[i]['skill'] = selected_skill

                        with col2:
                            proficiency = st.slider(
                                "Proficiency",
                                min_value=0,
                                max_value=100,
                                value=skill_item.get('proficiency', 50),
                                step=5,
                                key=f"proficiency_edit_{i}",
                                label_visibility="collapsed"
                            )
                            st.session_state.skills_list[i]['proficiency'] = proficiency

                        with col3:
                            if len(st.session_state.skills_list) > 1:
                                if st.button("🗑️", key=f"remove_edit_{i}", help="Remove skill"):
                                    skills_to_remove.append(i)

                        st.markdown('</div>', unsafe_allow_html=True)

                # Remove marked skills
                for i in reversed(skills_to_remove):
                    st.session_state.skills_list.pop(i)
                    st.rerun()

                # Action buttons
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("🚀 Re-analyze", type="primary"):
                        st.session_state.show_skill_editor = False
                        st.rerun()

                with col2:
                    if st.button("❌ Cancel Edit"):
                        st.session_state.show_skill_editor = False
                        st.rerun()

        # Export functionality
        st.markdown('<div class="sub-header">📤 Export Report</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📋 Copy Analysis Results"):
                skills_summary = ", ".join([f"{s['skill']}({s['proficiency']}%)" for s in skills_data if s['skill']])
                result_text = f"""
AI Career Compass Analysis Report

Career Type: {cluster_type}
Predicted Salary: ${predicted_salary:,.0f}
Salary Range: ${lower_bound:,.0f} - ${upper_bound:,.0f}
Years of Experience: {profile['years_experience']} years
Education: {profile['education_required']} ({profile.get('school_tier', 'Other')})

Skills Background: {skills_summary}

Core Strengths: {advice_data['strengths']}
Skill Gaps: {advice_data['gaps']}
Next Steps: {advice_data['next_steps']}
                """
                st.code(result_text)

        with col2:
            if st.button("🔄 Re-analyze"):
                st.session_state.analysis_complete = False
                st.rerun()

    # Author information footer
    st.markdown("""
    <div class="author-footer">
        <p><strong>🧭 AI Career Compass</strong></p>
        <p>Created by <a href="https://www.linkedin.com/in/meng-ni-felicia/" target="_blank">Meng Ni (Felicia)</a></p>
        <p><small>© 2024 AI Career Compass. Helping you find the right direction in your AI career path.</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()