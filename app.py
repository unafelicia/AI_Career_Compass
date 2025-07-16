import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import openai  # 暂时不用GPT API
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
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}


# 加载模型和编码器（这里需要你提供实际的模型文件）
@st.cache_resource
def load_models():
    try:
        # 这里需要你的实际模型文件路径
        salary_model = joblib.load('salary_prediction_model.pkl')  # 你的RF模型
        label_encoders = joblib.load('label_encoders.pkl')  # 你的编码器
        return salary_model, label_encoders
    except FileNotFoundError:
        st.error("模型文件未找到，请确保模型文件在正确路径")
        return None, None


# 技能聚类函数（基于你的代码）
def analyze_skills_cluster(skills_text):
    """基于技能文本进行聚类分析"""
    if not skills_text:
        return "未知类型", []

    # 模拟你的聚类逻辑
    skill_clusters = {
        "数据科学专家": ["python", "machine learning", "statistics", "data analysis", "pandas"],
        "AI工程师": ["tensorflow", "pytorch", "deep learning", "neural networks", "computer vision"],
        "数据工程师": ["sql", "spark", "hadoop", "etl", "databases", "aws"],
        "商业分析师": ["excel", "tableau", "powerbi", "business intelligence", "analytics"],
        "研究科学家": ["research", "phd", "publications", "mathematics", "algorithms"]
    }

    skills_lower = skills_text.lower()
    cluster_scores = {}

    for cluster_name, keywords in skill_clusters.items():
        score = sum(1 for keyword in keywords if keyword in skills_lower)
        cluster_scores[cluster_name] = score

    best_cluster = max(cluster_scores, key=cluster_scores.get)
    recommended_skills = skill_clusters[best_cluster]

    return best_cluster, recommended_skills


# 薪资预测函数
def predict_salary(user_data, salary_model, label_encoders):
    """基于用户数据预测薪资"""
    if not salary_model or not label_encoders:
        return 85000, 75000, 95000  # 默认值

    try:
        # 准备特征数据
        features = [
            'experience_level', 'employment_type', 'company_location',
            'company_size', 'employee_residence', 'remote_ratio',
            'education_required', 'years_experience', 'industry'
        ]

        # 创建预测数据
        pred_data = pd.DataFrame([user_data])

        # 应用编码器
        for col in pred_data.columns:
            if col in label_encoders and col != 'years_experience' and col != 'remote_ratio':
                try:
                    pred_data[col] = label_encoders[col].transform(pred_data[col])
                except ValueError:
                    # 如果遇到未见过的值，使用最常见的值
                    pred_data[col] = 0

        # 预测
        predicted_salary = salary_model.predict(pred_data[features])[0]

        # 计算置信区间（简化版）
        lower_bound = predicted_salary * 0.85
        upper_bound = predicted_salary * 1.15

        return predicted_salary, lower_bound, upper_bound

    except Exception as e:
        st.error(f"预测过程中出现错误: {str(e)}")
        return 85000, 75000, 95000


# 智能建议生成（基于规则，无需API）
def generate_career_advice(user_profile, cluster_type, salary_range):
    """生成个性化职业建议（基于规则系统）"""

    experience_level = user_profile['years_experience']
    education = user_profile['education_required']
    skills = user_profile['skills_text'].lower()

    # 基于经验水平的建议
    if experience_level <= 2:
        experience_advice = "作为初级从业者，重点是打好基础和积累实战经验"
        experience_tips = [
            "多做项目，建立作品集",
            "参与开源项目，提升协作能力",
            "考虑实习或entry-level职位"
        ]
    elif experience_level <= 5:
        experience_advice = "你正处于技能提升的关键期，可以开始专业化发展"
        experience_tips = [
            "选择1-2个专业方向深入",
            "考虑技术认证或进修",
            "开始承担更多责任"
        ]
    else:
        experience_advice = "作为资深从业者，可以考虑技术领导或专家路线"
        experience_tips = [
            "培养团队管理能力",
            "关注行业趋势和新技术",
            "考虑分享知识，建立影响力"
        ]

    # 基于职业类型的具体建议
    cluster_advice = {
        "数据科学专家": {
            "description": "你具备数据科学的核心技能，在分析和建模方面有优势",
            "strengths": ["数据处理能力", "统计分析技能", "机器学习基础"],
            "growth_areas": ["深度学习", "大数据处理", "业务理解"],
            "recommended_skills": ["TensorFlow/PyTorch", "Spark", "Docker", "AWS/GCP"],
            "career_paths": ["高级数据科学家", "ML工程师", "数据科学团队Lead"],
            "salary_potential": "随着经验增长，薪资可达$120K-180K"
        },
        "AI工程师": {
            "description": "你在AI技术实现方面很强，适合产品化和工程化工作",
            "strengths": ["深度学习框架", "模型部署", "算法实现"],
            "growth_areas": ["MLOps", "系统架构", "性能优化"],
            "recommended_skills": ["Kubernetes", "MLflow", "TensorFlow Serving", "CUDA"],
            "career_paths": ["Senior AI Engineer", "ML Platform Engineer", "AI Architect"],
            "salary_potential": "高级AI工程师薪资通常在$130K-200K"
        },
        "数据工程师": {
            "description": "你在数据基础设施方面有专长，是数据团队的重要支撑",
            "strengths": ["数据管道构建", "数据库管理", "ETL流程"],
            "growth_areas": ["实时数据处理", "云平台", "数据治理"],
            "recommended_skills": ["Kafka", "Airflow", "Snowflake", "dbt"],
            "career_paths": ["Senior Data Engineer", "Data Platform Lead", "Data Architect"],
            "salary_potential": "资深数据工程师薪资范围$110K-160K"
        },
        "商业分析师": {
            "description": "你在业务分析和数据洞察方面有天赋，桥接技术和业务",
            "strengths": ["业务理解", "数据可视化", "沟通表达"],
            "growth_areas": ["高级分析", "预测建模", "产品分析"],
            "recommended_skills": ["Python/R", "Advanced SQL", "A/B Testing", "Tableau"],
            "career_paths": ["Senior Business Analyst", "Product Analyst", "Analytics Manager"],
            "salary_potential": "高级商业分析师薪资通常在$90K-140K"
        },
        "研究科学家": {
            "description": "你在理论研究和创新方面有优势，适合前沿技术探索",
            "strengths": ["理论基础", "研究方法", "创新思维"],
            "growth_areas": ["工程实践", "产品化", "团队协作"],
            "recommended_skills": ["论文写作", "开源贡献", "技术演讲", "原型开发"],
            "career_paths": ["Principal Research Scientist", "Research Director", "CTO"],
            "salary_potential": "资深研究科学家薪资可达$150K-250K+"
        }
    }

    advice_data = cluster_advice.get(cluster_type, cluster_advice["数据科学专家"])

    # 基于薪资的市场分析
    salary_level = "高级" if salary_range[0] > 100000 else "中级" if salary_range[0] > 75000 else "入门"

    if salary_level == "入门":
        salary_advice = "你的薪资预测显示还有很大提升空间，建议重点提升核心技能"
    elif salary_level == "中级":
        salary_advice = "你的薪资水平在市场中位数左右，可以考虑专业化发展"
    else:
        salary_advice = "你的薪资预测较高，建议关注领导力和战略技能的培养"

    # 基于教育背景的建议
    education_advice = {
        "Bachelor": "考虑通过项目经验和认证来补充学历优势",
        "Master": "很好的学历基础，可以在专业领域深入发展",
        "PhD": "优秀的研究背景，可以考虑技术专家或管理路线"
    }

    # 生成最终建议
    advice = f"""
    ## 🎯 个性化职业发展建议

    ### 📊 你的职业画像：{cluster_type}
    {advice_data['description']}

    ### 💪 核心优势
    {chr(10).join(f"• {strength}" for strength in advice_data['strengths'])}

    ### 🎯 建议发展方向
    {chr(10).join(f"• {area}" for area in advice_data['growth_areas'])}

    ### 📚 推荐学习技能
    {chr(10).join(f"• {skill}" for skill in advice_data['recommended_skills'])}

    ### 🚀 职业发展路径
    {chr(10).join(f"• {path}" for path in advice_data['career_paths'])}

    ### 💰 薪资洞察
    预测薪资: **${salary_range[0]:,.0f}** (范围: ${salary_range[1]:,.0f} - ${salary_range[2]:,.0f})

    市场水平: **{salary_level}**级别

    {salary_advice}

    ### 🎓 基于你的背景
    **经验水平**: {experience_advice}

    **具体建议**:
    {chr(10).join(f"• {tip}" for tip in experience_tips)}

    **学历优势**: {education_advice.get(education, "继续学习是持续发展的关键")}

    ### 📈 未来展望
    {advice_data['salary_potential']}

    ### 🔥 立即行动建议
    1. **短期(1-3个月)**: 从推荐技能中选择1-2个开始学习
    2. **中期(3-6个月)**: 完成一个展示新技能的项目
    3. **长期(6-12个月)**: 根据职业路径调整求职策略

    *💡 提示: 这个分析基于你当前的技能和市场趋势，建议定期更新评估*
    """

    return advice


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

        # 技能信息
        st.subheader("技能背景")
        skills_text = st.text_area("描述你的技能(用逗号分隔)",
                                   placeholder="例如: Python, Machine Learning, SQL, Data Analysis")

        # 求职偏好
        st.subheader("求职偏好")
        experience_level = st.selectbox("经验级别", ["EN", "MI", "SE", "EX"])
        employment_type = st.selectbox("工作类型", ["FT", "PT", "CT", "FL"])
        company_location = st.selectbox("公司位置", ["United States", "United Kingdom", "Canada", "Germany", "Other"])
        company_size = st.selectbox("公司规模", ["S", "M", "L"])
        remote_ratio = st.selectbox("远程工作比例", [0, 50, 100])
        industry = st.selectbox("行业", ["Technology", "Finance", "Healthcare", "Media", "Retail", "Energy"])

        # 分析按钮
        if st.button("🚀 开始分析", type="primary"):
            # 保存用户数据
            st.session_state.user_profile = {
                'years_experience': years_experience,
                'education_required': education_level,
                'skills_text': skills_text,
                'experience_level': experience_level,
                'employment_type': employment_type,
                'company_location': company_location,
                'employee_residence': company_location,  # 简化处理
                'company_size': company_size,
                'remote_ratio': remote_ratio,
                'industry': industry
            }
            st.session_state.analysis_complete = True
            st.rerun()

    # 主页面：结果展示
    if st.session_state.analysis_complete:
        profile = st.session_state.user_profile

        # 技能聚类分析
        cluster_type, recommended_skills = analyze_skills_cluster(profile['skills_text'])

        # 薪资预测
        predicted_salary, lower_bound, upper_bound = predict_salary(
            profile, salary_model, label_encoders
        )

        # 结果展示
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
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 经验水平</h3>
                <h2>{profile['years_experience']} 年</h2>
                <p>{profile['education_required']}</p>
            </div>
            """, unsafe_allow_html=True)

        # 详细分析
        st.markdown('<div class="sub-header">📊 详细分析</div>', unsafe_allow_html=True)

        # 技能雷达图
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 技能画像")

            # 创建雷达图数据
            skills_data = {
                'Programming': 70 if 'python' in profile['skills_text'].lower() else 30,
                'Machine Learning': 80 if 'machine learning' in profile['skills_text'].lower() else 20,
                'Data Analysis': 75 if 'data' in profile['skills_text'].lower() else 25,
                'Statistics': 60 if 'statistics' in profile['skills_text'].lower() else 40,
                'Communication': 50 + (profile['years_experience'] * 5),
                'Domain Knowledge': 40 + (profile['years_experience'] * 3)
            }

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=list(skills_data.values()),
                theta=list(skills_data.keys()),
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

            fig = px.bar(
                x=list(comparison_data.keys()),
                y=list(comparison_data.values()),
                title="薪资水平对比",
                color=['lightblue', 'lightblue', 'lightblue', 'lightblue', 'darkblue'],
                color_discrete_map="identity"
            )

            fig.update_layout(
                xaxis_title="职业水平",
                yaxis_title="薪资 (USD)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # 智能建议系统
        st.markdown('<div class="sub-header">🧠 智能职业建议</div>', unsafe_allow_html=True)

        advice = generate_career_advice(
            profile, cluster_type, (predicted_salary, lower_bound, upper_bound)
        )

        st.markdown(f'<div class="insight-box">{advice}</div>', unsafe_allow_html=True)

        # 推荐技能
        st.markdown('<div class="sub-header">🎓 推荐学习技能</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**基于你的职业类型，推荐学习：**")
            for skill in recommended_skills[:3]:
                st.write(f"• {skill}")

        with col2:
            st.write("**高薪技能推荐：**")
            high_value_skills = ["Deep Learning", "MLOps", "Cloud Computing", "Data Engineering"]
            for skill in high_value_skills:
                st.write(f"• {skill}")

        # 导出功能
        st.markdown('<div class="sub-header">📤 导出报告</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📋 复制分析结果"):
                result_text = f"""
                AI Career Compass 分析报告

                职业类型: {cluster_type}
                预测薪资: ${predicted_salary:,.0f}
                薪资范围: ${lower_bound:,.0f} - ${upper_bound:,.0f}
                经验年限: {profile['years_experience']} 年
                学历: {profile['education_required']}

                技能背景: {profile['skills_text']}

                {advice}
                """
                st.code(result_text)

        with col2:
            if st.button("🔄 重新分析"):
                st.session_state.analysis_complete = False
                st.rerun()

    else:
        # 欢迎页面
        st.markdown("""
        ## 🚀 开始你的AI职业分析

        在左侧填写你的信息，我们将为你提供：

        - 🎯 **职业类型识别** - 基于技能聚类分析
        - 💰 **薪资预测** - 机器学习模型预测
        - 📊 **技能画像** - 可视化你的技能分布
        - 🧠 **智能建议** - 基于规则的个性化建议系统
        - 📈 **成长路径** - 针对性的技能提升建议

        ### 📋 使用指南
        1. 在左侧输入你的基本信息
        2. 详细描述你的技能背景
        3. 设置求职偏好
        4. 点击"开始分析"获得完整报告

        ### 🎯 适用人群
        - 准备转行AI/数据领域的人士
        - 1-3年经验的初级从业者
        - 对薪资和职业发展有疑问的求职者
        """)

        # 展示一些示例图表
        st.markdown("### 📊 示例分析")

        # 示例数据
        sample_data = {
            'Job Type': ['Data Scientist', 'ML Engineer', 'Data Analyst', 'Research Scientist'],
            'Average Salary': [95000, 110000, 70000, 125000],
            'Count': [450, 320, 600, 180]
        }

        fig = px.scatter(
            x=sample_data['Count'],
            y=sample_data['Average Salary'],
            size=sample_data['Count'],
            hover_name=sample_data['Job Type'],
            title="AI职位薪资分布示例",
            labels={'x': '职位数量', 'y': '平均薪资 (USD)'}
        )

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()