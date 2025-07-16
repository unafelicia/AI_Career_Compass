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

# 页面配置
st.set_page_config(
    page_title="🧭 AI Career Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
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

# 初始化session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}
if 'skills_list' not in st.session_state:
    st.session_state.skills_list = [{"skill": "", "proficiency": 50} for _ in range(3)]

# 预定义的技能列表
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

# 分级学习资源链接
LEARNING_RESOURCES = {
    "Python": {
        "beginner": [
            {"name": "Python官方教程", "url": "https://docs.python.org/3/tutorial/", "type": "文档"},
            {"name": "Automate the Boring Stuff", "url": "https://automatetheboringstuff.com/", "type": "在线书籍"},
            {"name": "Python for Everybody (Coursera)", "url": "https://www.coursera.org/specializations/python",
             "type": "课程"}
        ],
        "intermediate": [
            {"name": "Real Python Advanced Tutorials", "url": "https://realpython.com/tutorials/advanced/",
             "type": "教程"},
            {"name": "Python Tricks Book", "url": "https://realpython.com/python-tricks/", "type": "进阶书籍"},
            {"name": "Advanced Python Programming",
             "url": "https://www.udemy.com/course/python-beyond-the-basics-object-oriented-programming/",
             "type": "课程"}
        ],
        "advanced": [
            {"name": "Python Internals", "url": "https://github.com/python/cpython/tree/main/Doc", "type": "源码研究"},
            {"name": "High Performance Python",
             "url": "https://www.oreilly.com/library/view/high-performance-python/9781492055013/",
             "type": "专家级书籍"},
            {"name": "Python C API", "url": "https://docs.python.org/3/extending/", "type": "扩展开发"}
        ]
    },
    "Machine Learning": {
        "beginner": [
            {"name": "Andrew Ng ML Course", "url": "https://www.coursera.org/learn/machine-learning", "type": "课程"},
            {"name": "Scikit-learn User Guide", "url": "https://scikit-learn.org/stable/user_guide.html",
             "type": "文档"},
            {"name": "Kaggle Learn ML", "url": "https://www.kaggle.com/learn/intro-to-machine-learning",
             "type": "实战教程"}
        ],
        "intermediate": [
            {"name": "Hands-On ML with Scikit-Learn", "url": "https://github.com/ageron/handson-ml2",
             "type": "实战项目"},
            {"name": "Feature Engineering Course", "url": "https://www.coursera.org/learn/feature-engineering",
             "type": "专业课程"},
            {"name": "ML Engineering for Production",
             "url": "https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops",
             "type": "工程实践"}
        ],
        "advanced": [
            {"name": "Advanced ML Specialization", "url": "https://www.coursera.org/specializations/aml",
             "type": "高级专业化"},
            {"name": "ML Research Papers", "url": "https://paperswithcode.com/", "type": "前沿研究"},
            {"name": "Custom ML Algorithms", "url": "https://github.com/rushter/MLAlgorithms", "type": "算法实现"}
        ]
    },
    "Deep Learning": {
        "beginner": [
            {"name": "Deep Learning Specialization", "url": "https://www.coursera.org/specializations/deep-learning",
             "type": "课程"},
            {"name": "PyTorch Tutorials", "url": "https://pytorch.org/tutorials/", "type": "文档"},
            {"name": "Fast.ai Practical DL", "url": "https://course.fast.ai/", "type": "实用课程"}
        ],
        "intermediate": [
            {"name": "Advanced PyTorch", "url": "https://pytorch.org/tutorials/intermediate/", "type": "进阶教程"},
            {"name": "CS231n Stanford", "url": "http://cs231n.stanford.edu/", "type": "学术课程"},
            {"name": "Deep Learning Book", "url": "https://www.deeplearningbook.org/", "type": "理论基础"}
        ],
        "advanced": [
            {"name": "Transformer Architecture",
             "url": "https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf",
             "type": "论文研究"},
            {"name": "Advanced GAN Techniques", "url": "https://github.com/eriklindernoren/PyTorch-GAN",
             "type": "前沿实现"},
            {"name": "Neural Architecture Search", "url": "https://arxiv.org/abs/1808.05377", "type": "研究前沿"}
        ]
    },
    "SQL": {
        "beginner": [
            {"name": "SQLBolt Interactive", "url": "https://sqlbolt.com/", "type": "互动教程"},
            {"name": "W3Schools SQL", "url": "https://www.w3schools.com/sql/", "type": "基础教程"},
            {"name": "SQL for Data Science", "url": "https://www.coursera.org/learn/sql-for-data-science",
             "type": "课程"}
        ],
        "intermediate": [
            {"name": "Advanced SQL Techniques", "url": "https://mode.com/sql-tutorial/advanced/", "type": "进阶教程"},
            {"name": "SQL Performance Tuning", "url": "https://use-the-index-luke.com/", "type": "性能优化"},
            {"name": "Window Functions Deep Dive",
             "url": "https://www.postgresql.org/docs/current/tutorial-window.html", "type": "专题教程"}
        ],
        "advanced": [
            {"name": "Query Optimization", "url": "https://www.postgresql.org/docs/current/planner-optimizer.html",
             "type": "优化器原理"},
            {"name": "Database Internals", "url": "https://www.databass.dev/", "type": "数据库内核"},
            {"name": "Distributed SQL Systems",
             "url": "https://architecture-center.github.io/azure-architecture-center/data-guide/relational-data/",
             "type": "分布式架构"}
        ]
    },
    "Data Analysis": {
        "beginner": [
            {"name": "Pandas Getting Started", "url": "https://pandas.pydata.org/docs/getting_started/index.html",
             "type": "文档"},
            {"name": "Data Analysis with Python", "url": "https://www.coursera.org/learn/data-analysis-with-python",
             "type": "课程"},
            {"name": "Kaggle Data Cleaning", "url": "https://www.kaggle.com/learn/data-cleaning", "type": "实战"}
        ],
        "intermediate": [
            {"name": "Advanced Pandas", "url": "https://pandas.pydata.org/docs/user_guide/advanced.html",
             "type": "进阶文档"},
            {"name": "Statistical Data Analysis",
             "url": "https://www.coursera.org/specializations/statistics-with-python", "type": "统计分析"},
            {"name": "Time Series Analysis", "url": "https://www.kaggle.com/learn/time-series", "type": "专题分析"}
        ],
        "advanced": [
            {"name": "Big Data Analytics", "url": "https://spark.apache.org/docs/latest/sql-programming-guide.html",
             "type": "大数据分析"},
            {"name": "Advanced Statistics",
             "url": "https://online.stanford.edu/courses/stats200-introduction-statistical-inference",
             "type": "高级统计"},
            {"name": "Causal Inference", "url": "https://mixtape.scunning.com/", "type": "因果推断"}
        ]
    },
    "Cloud Computing": {
        "beginner": [
            {"name": "AWS Getting Started", "url": "https://aws.amazon.com/getting-started/", "type": "官方入门"},
            {"name": "Cloud Fundamentals", "url": "https://www.coursera.org/learn/introduction-to-cloud",
             "type": "基础课程"},
            {"name": "Azure Fundamentals", "url": "https://docs.microsoft.com/en-us/learn/paths/azure-fundamentals/",
             "type": "微软认证"}
        ],
        "intermediate": [
            {"name": "AWS Solutions Architect",
             "url": "https://aws.amazon.com/certification/certified-solutions-architect-associate/",
             "type": "专业认证"},
            {"name": "Kubernetes Deep Dive", "url": "https://kubernetes.io/docs/tutorials/", "type": "容器编排"},
            {"name": "DevOps with Cloud",
             "url": "https://www.coursera.org/specializations/devops-cloud-and-agile-foundations", "type": "运维实践"}
        ],
        "advanced": [
            {"name": "Cloud Architecture Patterns", "url": "https://docs.microsoft.com/en-us/azure/architecture/",
             "type": "架构设计"},
            {"name": "Multi-Cloud Strategy", "url": "https://cloud.google.com/architecture/framework",
             "type": "多云策略"},
            {"name": "Serverless Computing", "url": "https://martinfowler.com/articles/serverless.html",
             "type": "无服务器"}
        ]
    },
    "MLOps": {
        "beginner": [
            {"name": "MLOps Fundamentals", "url": "https://ml-ops.org/content/motivation", "type": "概念入门"},
            {"name": "MLflow Tutorial", "url": "https://mlflow.org/docs/latest/tutorials-and-examples/index.html",
             "type": "工具教程"},
            {"name": "ML Pipelines Intro",
             "url": "https://www.coursera.org/learn/machine-learning-engineering-for-production-mlops",
             "type": "流水线"}
        ],
        "intermediate": [
            {"name": "Kubeflow Pipelines", "url": "https://www.kubeflow.org/docs/components/pipelines/",
             "type": "生产部署"},
            {"name": "Model Monitoring",
             "url": "https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/",
             "type": "模型监控"},
            {"name": "A/B Testing for ML",
             "url": "https://booking.ai/how-booking-com-increases-the-power-of-online-experiments-with-machine-learning-25aa102df618",
             "type": "实验设计"}
        ],
        "advanced": [
            {"name": "ML System Design", "url": "https://github.com/chiphuyen/machine-learning-systems-design",
             "type": "系统设计"},
            {"name": "Feature Stores", "url": "https://www.tecton.ai/blog/what-is-a-feature-store/",
             "type": "特征工程"},
            {"name": "Real-time ML", "url": "https://huyenchip.com/2020/12/27/real-time-machine-learning.html",
             "type": "实时系统"}
        ]
    }
}

# 学校类型权重配置
SCHOOL_WEIGHTS = {
    "985/QS Top 50": 1.2,
    "211/QS Top 200": 1.05,
    "其他": 1.0
}

# 行业权重配置
INDUSTRY_WEIGHTS = {
    "Technology": 1.1,
    "Finance": 1.2,
    "Healthcare": 1.0,
    "Media": 0.9,
    "Retail": 0.8,
    "Energy": 0.95,
    "其他": 1.0
}


# 加载模型和编码器
@st.cache_resource
def load_models():
    try:
        salary_model = joblib.load('salary_prediction_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        return salary_model, label_encoders
    except FileNotFoundError:
        st.error("模型文件未找到，请确保模型文件在正确路径")
        return None, None


# 改进的技能聚类函数
def analyze_skills_cluster(skills_data):
    """基于技能组合和熟练度进行聚类分析"""
    if not skills_data or not any(skill['skill'] for skill in skills_data):
        return "通用型人才", []

    # 定义职业类型的技能特征
    skill_patterns = {
        "数据科学专家": {
            "core_skills": ["python", "machine learning", "statistics", "data analysis"],
            "bonus_skills": ["pandas", "numpy", "scikit-learn", "jupyter"],
            "min_core_count": 2,  # 至少需要2个核心技能
            "weight_factor": 1.0
        },
        "AI工程师": {
            "core_skills": ["deep learning", "tensorflow", "pytorch", "neural networks"],
            "bonus_skills": ["computer vision", "nlp", "python", "machine learning"],
            "min_core_count": 2,
            "weight_factor": 1.1
        },
        "数据工程师": {
            "core_skills": ["sql", "spark", "hadoop", "etl"],
            "bonus_skills": ["aws", "azure", "docker", "kubernetes", "python"],
            "min_core_count": 2,
            "weight_factor": 1.05
        },
        "商业分析师": {
            "core_skills": ["excel", "tableau", "power bi", "business intelligence"],
            "bonus_skills": ["sql", "analytics", "data visualization"],
            "min_core_count": 1,
            "weight_factor": 0.95
        }
    }

    user_skills = []
    total_proficiency = 0

    # 提取用户技能和熟练度
    for skill_item in skills_data:
        if skill_item['skill']:
            skill_lower = skill_item['skill'].lower()
            proficiency = skill_item['proficiency']
            user_skills.append((skill_lower, proficiency))
            total_proficiency += proficiency

    if not user_skills:
        return "通用型人才", PREDEFINED_SKILLS[:5]

    # 计算每个职业类型的匹配分数
    cluster_scores = {}

    for cluster_name, pattern in skill_patterns.items():
        core_matches = 0
        core_proficiency_sum = 0
        bonus_score = 0

        # 检查核心技能匹配
        for skill_lower, proficiency in user_skills:
            for core_skill in pattern["core_skills"]:
                if core_skill in skill_lower:
                    core_matches += 1
                    core_proficiency_sum += proficiency
                    break

            # 检查奖励技能
            for bonus_skill in pattern["bonus_skills"]:
                if bonus_skill in skill_lower:
                    bonus_score += proficiency * 0.5
                    break

        # 只有满足最少核心技能数量要求才考虑该职业类型
        if core_matches >= pattern["min_core_count"]:
            # 计算综合分数
            core_score = (core_proficiency_sum / max(core_matches, 1)) * core_matches
            total_score = (core_score + bonus_score) * pattern["weight_factor"]
            cluster_scores[cluster_name] = total_score
        else:
            cluster_scores[cluster_name] = 0

    # 如果没有明确的职业类型匹配，返回通用型
    if not cluster_scores or max(cluster_scores.values()) == 0:
        return "通用型人才", PREDEFINED_SKILLS[:5]

    # 返回得分最高的职业类型
    best_cluster = max(cluster_scores, key=cluster_scores.get)
    recommended_skills = (skill_patterns[best_cluster]["core_skills"] +
                          skill_patterns[best_cluster]["bonus_skills"])[:5]

    return best_cluster, recommended_skills


# 薪资预测函数
def predict_salary(user_data, skills_data, salary_model, label_encoders):
    """基于用户数据预测薪资，包含学校和技能权重"""
    if not salary_model or not label_encoders:
        base_salary = calculate_base_salary(user_data, skills_data)
        return base_salary, base_salary * 0.85, base_salary * 1.15

    try:
        base_salary = calculate_base_salary(user_data, skills_data)

        # 应用学校权重
        school_weight = SCHOOL_WEIGHTS.get(user_data.get('school_tier', '其他'), 1.0)

        # 应用行业权重
        industry_weight = INDUSTRY_WEIGHTS.get(user_data.get('industry', '其他'), 1.0)

        # 计算技能权重
        skill_weight = calculate_skill_weight(skills_data)

        # 综合计算
        predicted_salary = base_salary * school_weight * industry_weight * skill_weight

        # 计算置信区间
        lower_bound = predicted_salary * 0.85
        upper_bound = predicted_salary * 1.15

        return predicted_salary, lower_bound, upper_bound

    except Exception as e:
        st.error(f"预测过程中出现错误: {str(e)}")
        base_salary = calculate_base_salary(user_data, skills_data)
        return base_salary, base_salary * 0.85, base_salary * 1.15


def calculate_base_salary(user_data, skills_data):
    """计算基础薪资"""
    base_salaries = {
        "入门级 (0-2年)": 70000,
        "中级 (2-5年)": 95000,
        "高级 (5-10年)": 130000,
        "专家级 (10年以上)": 180000
    }

    experience_level = user_data.get('experience_level', '中级 (2-5年)')
    base = base_salaries.get(experience_level, 95000)

    # 地区调整
    location_multipliers = {
        "United States": 1.0, "United Kingdom": 0.8, "Canada": 0.85,
        "Germany": 0.75, "Netherlands": 0.8, "China": 0.6, "Singapore": 0.9
    }

    location = user_data.get('company_location', 'United States')
    base *= location_multipliers.get(location, 0.8)

    # 公司规模调整
    size_multipliers = {
        "小型公司 (<50人)": 0.9, "中型公司 (50-250人)": 1.0, "大型公司 (>250人)": 1.15
    }

    company_size = user_data.get('company_size', '中型公司 (50-250人)')
    base *= size_multipliers.get(company_size, 1.0)

    # 远程工作调整
    remote_multipliers = {
        "0% (完全现场办公)": 1.0, "25% (偶尔远程)": 1.02, "50% (混合办公)": 1.05,
        "75% (主要远程)": 1.08, "100% (完全远程)": 1.1
    }

    remote_ratio = user_data.get('remote_ratio', '50% (混合办公)')
    base *= remote_multipliers.get(remote_ratio, 1.0)

    return int(base)


def calculate_skill_weight(skills_data):
    """根据技能数量和熟练度计算权重"""
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


# 生成职业建议
def generate_career_advice(user_profile, cluster_type, salary_range, skills_data):
    """生成个性化职业建议"""
    advice_templates = {
        "数据科学专家": {
            "strengths": "你在数据分析和机器学习方面有很好的基础",
            "gaps": "建议加强深度学习和大数据处理技能",
            "next_steps": "考虑学习TensorFlow/PyTorch，获得AWS认证",
            "icon": "📊"
        },
        "AI工程师": {
            "strengths": "你在AI技术栈方面有很强的技术能力",
            "gaps": "建议加强产品化和工程实践经验",
            "next_steps": "参与开源项目，学习MLOps和模型部署",
            "icon": "🤖"
        },
        "数据工程师": {
            "strengths": "你在数据基础设施方面有很好的技能",
            "gaps": "建议学习更多云平台和实时数据处理技术",
            "next_steps": "深入学习Kafka、Kubernetes等技术",
            "icon": "🔧"
        },
        "商业分析师": {
            "strengths": "你在业务理解和数据可视化方面很有优势",
            "gaps": "建议加强编程技能和统计分析能力",
            "next_steps": "学习Python/R，掌握高级分析方法",
            "icon": "📈"
        },
        "通用型人才": {
            "strengths": "你具备多方面的技能基础",
            "gaps": "建议专注某个细分领域深入发展",
            "next_steps": "选择感兴趣的方向，系统性提升专业技能",
            "icon": "🎯"
        }
    }

    template = advice_templates.get(cluster_type, advice_templates["通用型人才"])

    return template


# 紧凑型技能输入组件
def skill_input_component():
    """紧凑型技能输入组件"""
    with st.expander("🎯 技能背景配置", expanded=True):
        st.write("选择你的技能并评估熟练程度：")

        # 添加技能按钮
        if st.button("➕ 添加技能", key="add_skill_btn"):
            st.session_state.skills_list.append({"skill": "", "proficiency": 50})
            st.rerun()

        # 技能输入行（紧凑布局）
        skills_to_remove = []
        for i, skill_item in enumerate(st.session_state.skills_list):
            with st.container():
                st.markdown(f'<div class="skill-compact-row">', unsafe_allow_html=True)

                col1, col2, col3 = st.columns([4, 3, 1])

                with col1:
                    skill_options = [""] + PREDEFINED_SKILLS + ["自定义..."]
                    current_skill = skill_item.get('skill', '')

                    if current_skill and current_skill not in PREDEFINED_SKILLS:
                        skill_options.insert(-1, current_skill)

                    selected_skill = st.selectbox(
                        f"技能 {i + 1}",
                        options=skill_options,
                        index=skill_options.index(current_skill) if current_skill in skill_options else 0,
                        key=f"skill_{i}",
                        label_visibility="collapsed"
                    )

                    if selected_skill == "自定义...":
                        custom_skill = st.text_input(
                            "",
                            value=current_skill if current_skill not in PREDEFINED_SKILLS else "",
                            key=f"custom_skill_{i}",
                            placeholder="输入自定义技能"
                        )
                        st.session_state.skills_list[i]['skill'] = custom_skill
                    else:
                        st.session_state.skills_list[i]['skill'] = selected_skill

                with col2:
                    proficiency = st.slider(
                        "熟练程度",
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
                        if st.button("🗑️", key=f"remove_{i}", help="删除技能"):
                            skills_to_remove.append(i)

                st.markdown('</div>', unsafe_allow_html=True)

        # 移除标记的技能
        for i in reversed(skills_to_remove):
            st.session_state.skills_list.pop(i)
            st.rerun()


# 显示学习资源（支持分级）
def display_learning_resources(skills, user_skills_data):
    """显示分级学习资源卡片"""
    st.markdown('<div class="sub-header">🎓 推荐学习技能与资源</div>', unsafe_allow_html=True)

    # 创建用户技能熟练度字典
    user_proficiency = {}
    for skill_item in user_skills_data:
        if skill_item['skill']:
            user_proficiency[skill_item['skill']] = skill_item['proficiency']

    # 确保至少显示4个技能的学习资源
    display_skills = []

    # 优先显示推荐技能中有资源的技能
    for skill in skills:
        if skill.title() in LEARNING_RESOURCES:
            display_skills.append(skill.title())
        elif skill.lower() in [k.lower() for k in LEARNING_RESOURCES.keys()]:
            for resource_key in LEARNING_RESOURCES.keys():
                if skill.lower() == resource_key.lower():
                    display_skills.append(resource_key)
                    break

    # 如果推荐技能中的资源不足4个，补充热门技能
    popular_skills = ["Python", "Machine Learning", "SQL", "Data Analysis", "Deep Learning", "Cloud Computing"]
    for skill in popular_skills:
        if len(display_skills) >= 4:
            break
        if skill not in display_skills:
            display_skills.append(skill)

    # 确保至少有4个技能显示
    if len(display_skills) < 4:
        display_skills = ["Python", "Machine Learning", "SQL", "Data Analysis"]

    # 显示前4个技能的学习资源
    for skill in display_skills[:4]:
        if skill in LEARNING_RESOURCES:
            # 根据用户熟练度确定资源级别
            proficiency = user_proficiency.get(skill, 0)

            if proficiency >= 80:
                level = "advanced"
                level_name = "高级"
                level_color = "#e74c3c"
            elif proficiency >= 60:
                level = "intermediate"
                level_name = "进阶"
                level_color = "#f39c12"
            else:
                level = "beginner"
                level_name = "入门"
                level_color = "#27ae60"

            st.markdown(
                f"### 📚 {skill} <span style='color: {level_color}; font-size: 0.8em;'>({level_name}级资源)</span>",
                unsafe_allow_html=True)

            # 显示对应级别的资源
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
            # 如果没有具体资源，显示通用学习建议
            st.markdown(f"### 📚 {skill}")
            st.markdown(f"""
            <div class="skill-resource-card">
                <strong>推荐学习 {skill}</strong><br>
                <small>建议通过在线课程、官方文档和实战项目来学习</small>
            </div>
            """, unsafe_allow_html=True)


# 主界面
def main():
    st.markdown('<div class="main-header">🧭 AI Career Compass</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 3rem;">AI求职助手：技能画像、薪资预测、成长建议</div>',
        unsafe_allow_html=True)

    # 加载模型
    salary_model, label_encoders = load_models()

    # 侧边栏：用户输入
    with st.sidebar:
        st.header("📝 个人信息输入")

        # 基本信息
        st.subheader("基本背景")
        years_experience = st.slider("工作经验年限", 0, 15, 3)

        education_level = st.selectbox("学历水平", ["Bachelor", "Master", "PhD"])
        school_tier = st.selectbox("学校层次", ["985/QS Top 50", "211/QS Top 200", "其他"])

        # 求职偏好
        st.subheader("求职偏好")
        experience_level = st.selectbox("经验级别",
                                        ["入门级 (0-2年)", "中级 (2-5年)", "高级 (5-10年)", "专家级 (10年以上)"])
        employment_type = st.selectbox("工作类型", ["全职", "兼职", "合同工", "自由职业"])
        company_location = st.selectbox("公司位置",
                                        ["United States", "United Kingdom", "Canada", "Germany", "Netherlands", "China",
                                         "Singapore", "Other"])
        company_size = st.selectbox("公司规模", ["小型公司 (<50人)", "中型公司 (50-250人)", "大型公司 (>250人)"])
        remote_ratio = st.selectbox("远程工作比例",
                                    ["0% (完全现场办公)", "25% (偶尔远程)", "50% (混合办公)", "75% (主要远程)",
                                     "100% (完全远程)"])
        industry = st.selectbox("行业", ["Technology", "Finance", "Healthcare", "Media", "Retail", "Energy", "其他"])

        # 分析按钮
        if st.button("🚀 开始分析", type="primary"):
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

    # 主页面内容
    if not st.session_state.analysis_complete:
        # 技能输入界面（紧凑版）
        skill_input_component()

        # 欢迎信息
        st.markdown("""
        ## 🚀 开始你的AI职业分析

        填写左侧的基本信息和上方的技能背景，我们将为你提供：

        - 🎯 **智能职业分类** - 基于技能组合的专业聚类分析
        - 💰 **精准薪资预测** - 考虑学校背景、技能熟练度的薪资模型
        - 📊 **可视化技能画像** - 雷达图展示你的技能分布
        - 🧠 **个性化职业建议** - AI驱动的发展建议
        - 🎓 **学习资源推荐** - 精选的技能提升资源链接
        """)

    else:
        # 结果展示页面
        profile = st.session_state.user_profile
        skills_data = st.session_state.skills_list

        # 技能聚类分析
        cluster_type, recommended_skills = analyze_skills_cluster(skills_data)

        # 薪资预测
        predicted_salary, lower_bound, upper_bound = predict_salary(profile, skills_data, salary_model, label_encoders)

        # 三个等宽指标卡片
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>👤 职业类型</h3>
                <h2>{cluster_type}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 预测薪资</h3>
                <h2>${predicted_salary:,.0f}</h2>
                <p>${lower_bound:,.0f} - ${upper_bound:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            school_display = profile.get('school_tier', '其他')
            weight_info = f"权重: {SCHOOL_WEIGHTS.get(school_display, 1.0):.2f}"
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎓 学历背景</h3>
                <h2>{profile['education_required']}</h2>
                <p>{school_display}</p>
                <small>{weight_info}</small>
            </div>
            """, unsafe_allow_html=True)

        # 详细分析
        st.markdown('<div class="sub-header">📊 详细分析</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 技能画像")

            # 恢复雷达图显示
            if skills_data and any(skill['skill'] for skill in skills_data):
                # 创建雷达图数据
                skills_dict = {}
                for skill_item in skills_data:
                    if skill_item['skill']:
                        skills_dict[skill_item['skill']] = skill_item['proficiency']

                # 补充常见技能维度
                radar_skills = ['Programming', 'Machine Learning', 'Data Analysis', 'Statistics', 'Communication',
                                'Domain Knowledge']
                radar_values = []

                for radar_skill in radar_skills:
                    # 根据用户技能匹配计算分数
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
                        # 基于经验年限估算
                        score = min(50 + (profile['years_experience'] * 5), 100)
                    elif radar_skill == 'Domain Knowledge':
                        # 基于经验年限和行业估算
                        score = min(40 + (profile['years_experience'] * 4), 100)

                    radar_values.append(score)

                # 创建雷达图
                fig = go.Figure()

                fig.add_trace(go.Scatterpolar(
                    r=radar_values,
                    theta=radar_skills,
                    fill='toself',
                    name='你的技能',
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
                st.info("请添加技能信息以查看技能画像")

        with col2:
            st.subheader("💼 市场对比")

            # 薪资对比图
            comparison_data = {
                'Entry Level': 65000,
                'Mid Level': 85000,
                'Senior Level': 120000,
                'Expert Level': 160000,
                '你的预测': predicted_salary
            }

            colors = ['lightblue'] * 4 + ['darkblue']

            fig = px.bar(
                x=list(comparison_data.keys()),
                y=list(comparison_data.values()),
                title="薪资水平对比",
                color=colors,
                color_discrete_map="identity"
            )

            fig.update_layout(
                xaxis_title="职业水平",
                yaxis_title="薪资 (USD)",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        # AI职业建议 - 卡片式布局
        st.markdown('<div class="sub-header">🧠 AI职业建议</div>', unsafe_allow_html=True)

        advice_data = generate_career_advice(profile, cluster_type, (predicted_salary, lower_bound, upper_bound),
                                             skills_data)

        # 建议卡片
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="advice-card">
                <div class="advice-header">
                    {advice_data['icon']} 核心优势
                </div>
                <div class="advice-content">
                    {advice_data['strengths']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="advice-card">
                <div class="advice-header">
                    🎯 发展计划
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
                    📈 技能补强
                </div>
                <div class="advice-content">
                    {advice_data['gaps']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 技能分析
            skill_analysis = ""
            if skills_data and any(skill['skill'] for skill in skills_data):
                high_skills = [s for s in skills_data if s['skill'] and s['proficiency'] >= 80]
                medium_skills = [s for s in skills_data if s['skill'] and 50 <= s['proficiency'] < 80]
                low_skills = [s for s in skills_data if s['skill'] and s['proficiency'] < 50]

                if high_skills:
                    skill_analysis += f"<strong>专长技能:</strong> {', '.join([s['skill'] for s in high_skills])}<br>"
                if low_skills:
                    skill_analysis += f"<strong>待提升:</strong> {', '.join([s['skill'] for s in low_skills])}"

            st.markdown(f"""
            <div class="advice-card">
                <div class="advice-header">
                    💰 薪资洞察
                </div>
                <div class="advice-content">
                    预测薪资: ${predicted_salary:,.0f} (${lower_bound:,.0f} - ${upper_bound:,.0f})<br>
                    市场水平: {'较高' if predicted_salary > 90000 else '中等' if predicted_salary > 70000 else '入门'}水平<br>
                    {skill_analysis}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 学习资源推荐
        display_learning_resources(recommended_skills, skills_data)

        # 编辑技能按钮
        st.markdown('<div class="sub-header">⚙️ 调整设置</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✏️ 编辑技能配置", type="secondary"):
                st.session_state.show_skill_editor = True
                st.rerun()

        # 技能编辑器
        if st.session_state.get('show_skill_editor', False):
            st.markdown("---")
            st.markdown("### ✏️ 技能配置编辑")

            # 重用技能输入组件
            with st.container():
                st.write("调整你的技能并重新评估熟练程度：")

                # 添加技能按钮
                if st.button("➕ 添加技能", key="add_skill_btn_edit"):
                    st.session_state.skills_list.append({"skill": "", "proficiency": 50})
                    st.rerun()

                # 技能输入行
                skills_to_remove = []
                for i, skill_item in enumerate(st.session_state.skills_list):
                    with st.container():
                        st.markdown(f'<div class="skill-compact-row">', unsafe_allow_html=True)

                        col1, col2, col3 = st.columns([4, 3, 1])

                        with col1:
                            skill_options = [""] + PREDEFINED_SKILLS + ["自定义..."]
                            current_skill = skill_item.get('skill', '')

                            if current_skill and current_skill not in PREDEFINED_SKILLS:
                                skill_options.insert(-1, current_skill)

                            selected_skill = st.selectbox(
                                f"技能 {i + 1}",
                                options=skill_options,
                                index=skill_options.index(current_skill) if current_skill in skill_options else 0,
                                key=f"skill_edit_{i}",
                                label_visibility="collapsed"
                            )

                            if selected_skill == "自定义...":
                                custom_skill = st.text_input(
                                    "",
                                    value=current_skill if current_skill not in PREDEFINED_SKILLS else "",
                                    key=f"custom_skill_edit_{i}",
                                    placeholder="输入自定义技能"
                                )
                                st.session_state.skills_list[i]['skill'] = custom_skill
                            else:
                                st.session_state.skills_list[i]['skill'] = selected_skill

                        with col2:
                            proficiency = st.slider(
                                "熟练程度",
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
                                if st.button("🗑️", key=f"remove_edit_{i}", help="删除技能"):
                                    skills_to_remove.append(i)

                        st.markdown('</div>', unsafe_allow_html=True)

                # 移除标记的技能
                for i in reversed(skills_to_remove):
                    st.session_state.skills_list.pop(i)
                    st.rerun()

                # 操作按钮
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("🚀 重新分析", type="primary"):
                        st.session_state.show_skill_editor = False
                        st.rerun()

                with col2:
                    if st.button("❌ 取消编辑"):
                        st.session_state.show_skill_editor = False
                        st.rerun()

        # 导出功能
        st.markdown('<div class="sub-header">📤 导出报告</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📋 复制分析结果"):
                skills_summary = ", ".join([f"{s['skill']}({s['proficiency']}%)" for s in skills_data if s['skill']])
                result_text = f"""
AI Career Compass 分析报告

职业类型: {cluster_type}
预测薪资: ${predicted_salary:,.0f}
薪资范围: ${lower_bound:,.0f} - ${upper_bound:,.0f}
经验年限: {profile['years_experience']} 年
学历: {profile['education_required']} ({profile.get('school_tier', '其他')})

技能背景: {skills_summary}

核心优势: {advice_data['strengths']}
技能差距: {advice_data['gaps']}
下一步行动: {advice_data['next_steps']}
                """
                st.code(result_text)

        with col2:
            if st.button("🔄 重新分析"):
                st.session_state.analysis_complete = False
                st.rerun()

    # 作者信息footer
    st.markdown("""
    <div class="author-footer">
        <p><strong>🧭 AI Career Compass</strong></p>
        <p>Created by <a href="https://www.linkedin.com/in/meng-ni-felicia/" target="_blank">Meng Ni (Felicia)</a></p>
        <p><small>© 2024 AI Career Compass. 帮助你在AI职业路径上找到正确方向。</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()