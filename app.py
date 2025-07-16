import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import openai
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
    .skill-row {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
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
    "其他": 1.0  # 使用平均值
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


# 技能聚类函数（基于技能和熟练度）
def analyze_skills_cluster(skills_data):
    """基于技能和熟练度进行聚类分析"""
    if not skills_data or not any(skill['skill'] for skill in skills_data):
        return "未知类型", []

    # 计算加权技能分数
    skill_clusters = {
        "数据科学专家": {
            "keywords": ["python", "machine learning", "statistics", "data analysis", "pandas", "numpy"],
            "weight_factor": 1.0
        },
        "AI工程师": {
            "keywords": ["tensorflow", "pytorch", "deep learning", "neural networks", "computer vision", "nlp"],
            "weight_factor": 1.1
        },
        "数据工程师": {
            "keywords": ["sql", "spark", "hadoop", "etl", "aws", "azure", "docker", "kubernetes"],
            "weight_factor": 1.05
        },
        "商业分析师": {
            "keywords": ["excel", "tableau", "power bi", "business intelligence", "analytics", "visualization"],
            "weight_factor": 0.95
        }
    }

    cluster_scores = {}
    total_proficiency = 0

    for cluster_name, cluster_info in skill_clusters.items():
        weighted_score = 0
        for skill_item in skills_data:
            if skill_item['skill']:
                skill_lower = skill_item['skill'].lower()
                proficiency = skill_item['proficiency']

                for keyword in cluster_info['keywords']:
                    if keyword in skill_lower:
                        weighted_score += (proficiency / 100) * cluster_info['weight_factor']

                total_proficiency += proficiency

        cluster_scores[cluster_name] = weighted_score

    if not cluster_scores or max(cluster_scores.values()) == 0:
        return "通用型人才", PREDEFINED_SKILLS[:5]

    best_cluster = max(cluster_scores, key=cluster_scores.get)
    recommended_skills = skill_clusters[best_cluster]['keywords']

    return best_cluster, recommended_skills


# 薪资预测函数（包含学校和技能权重）
def predict_salary(user_data, skills_data, salary_model, label_encoders):
    """基于用户数据预测薪资，包含学校和技能权重"""
    if not salary_model or not label_encoders:
        base_salary = calculate_base_salary(user_data, skills_data)
        return base_salary, base_salary * 0.85, base_salary * 1.15

    try:
        # 基础薪资预测逻辑
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
    # 基础薪资根据经验级别
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
        "United States": 1.0,
        "United Kingdom": 0.8,
        "Canada": 0.85,
        "Germany": 0.75,
        "Netherlands": 0.8,
        "China": 0.6,
        "Singapore": 0.9
    }

    location = user_data.get('company_location', 'United States')
    base *= location_multipliers.get(location, 0.8)

    # 公司规模调整
    size_multipliers = {
        "小型公司 (<50人)": 0.9,
        "中型公司 (50-250人)": 1.0,
        "大型公司 (>250人)": 1.15
    }

    company_size = user_data.get('company_size', '中型公司 (50-250人)')
    base *= size_multipliers.get(company_size, 1.0)

    # 远程工作调整
    remote_multipliers = {
        "0% (完全现场办公)": 1.0,
        "25% (偶尔远程)": 1.02,
        "50% (混合办公)": 1.05,
        "75% (主要远程)": 1.08,
        "100% (完全远程)": 1.1
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

    # 平均熟练度转换为权重
    avg_proficiency = total_score / skill_count
    skill_weight = 0.8 + (avg_proficiency / 100) * 0.4  # 0.8-1.2之间

    # 技能数量奖励
    skill_count_bonus = min(skill_count * 0.02, 0.1)  # 最多10%奖励

    return skill_weight + skill_count_bonus


# GPT建议生成
def generate_career_advice(user_profile, cluster_type, salary_range, skills_data):
    """生成个性化职业建议"""
    advice_templates = {
        "数据科学专家": {
            "strengths": "你在数据分析和机器学习方面有很好的基础",
            "gaps": "建议加强深度学习和大数据处理技能",
            "next_steps": "考虑学习TensorFlow/PyTorch，获得AWS认证"
        },
        "AI工程师": {
            "strengths": "你在AI技术栈方面有很强的技术能力",
            "gaps": "建议加强产品化和工程实践经验",
            "next_steps": "参与开源项目，学习MLOps和模型部署"
        },
        "数据工程师": {
            "strengths": "你在数据基础设施方面有很好的技能",
            "gaps": "建议学习更多云平台和实时数据处理技术",
            "next_steps": "深入学习Kafka、Kubernetes等技术"
        },
        "商业分析师": {
            "strengths": "你在业务理解和数据可视化方面很有优势",
            "gaps": "建议加强编程技能和统计分析能力",
            "next_steps": "学习Python/R，掌握高级分析方法"
        },
        "通用型人才": {
            "strengths": "你具备多方面的技能基础",
            "gaps": "建议专注某个细分领域深入发展",
            "next_steps": "选择感兴趣的方向，系统性提升专业技能"
        }
    }

    template = advice_templates.get(cluster_type, advice_templates["通用型人才"])

    # 分析技能熟练度
    skill_analysis = ""
    if skills_data:
        high_skills = [s for s in skills_data if s['skill'] and s['proficiency'] >= 80]
        medium_skills = [s for s in skills_data if s['skill'] and 50 <= s['proficiency'] < 80]
        low_skills = [s for s in skills_data if s['skill'] and s['proficiency'] < 50]

        if high_skills:
            skill_analysis += f"**核心技能**: {', '.join([s['skill'] for s in high_skills])}\n\n"
        if medium_skills:
            skill_analysis += f"**发展中技能**: {', '.join([s['skill'] for s in medium_skills])}\n\n"
        if low_skills:
            skill_analysis += f"**待提升技能**: {', '.join([s['skill'] for s in low_skills])}\n\n"

    advice = f"""
    ## 🎯 个性化职业建议

    **你的职业类型**: {cluster_type}

    **核心优势**: {template['strengths']}

    **技能差距**: {template['gaps']}

    **下一步行动**: {template['next_steps']}

    {skill_analysis}

    **薪资洞察**: 基于你的背景，预测薪资范围在 ${salary_range[1]:,.0f} - ${salary_range[2]:,.0f}，
    中位数约为 ${salary_range[0]:,.0f}。这个水平在同类型人才中属于{'较高' if salary_range[0] > 90000 else '中等' if salary_range[0] > 70000 else '入门'}水平。
    """

    return advice


# 技能输入组件
def skill_input_component():
    """技能输入组件"""
    st.subheader("🎯 技能背景")
    st.write("选择你的技能并评估熟练程度：")

    # 添加技能按钮
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕ 添加技能"):
            st.session_state.skills_list.append({"skill": "", "proficiency": 50})
            st.rerun()

    # 技能输入行
    skills_to_remove = []
    for i, skill_item in enumerate(st.session_state.skills_list):
        with st.container():
            st.markdown(f'<div class="skill-row">', unsafe_allow_html=True)

            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                # 技能选择（可以输入自定义技能）
                skill_options = [""] + PREDEFINED_SKILLS + ["自定义..."]
                current_skill = skill_item.get('skill', '')

                if current_skill and current_skill not in PREDEFINED_SKILLS:
                    skill_options.insert(-1, current_skill)

                selected_skill = st.selectbox(
                    f"技能 {i + 1}",
                    options=skill_options,
                    index=skill_options.index(current_skill) if current_skill in skill_options else 0,
                    key=f"skill_{i}"
                )

                # 如果选择自定义，显示文本输入
                if selected_skill == "自定义...":
                    custom_skill = st.text_input(
                        "输入自定义技能",
                        value=current_skill if current_skill not in PREDEFINED_SKILLS else "",
                        key=f"custom_skill_{i}"
                    )
                    st.session_state.skills_list[i]['skill'] = custom_skill
                else:
                    st.session_state.skills_list[i]['skill'] = selected_skill

            with col2:
                # 熟练程度滑块
                proficiency = st.slider(
                    "熟练程度",
                    min_value=0,
                    max_value=100,
                    value=skill_item.get('proficiency', 50),
                    step=5,
                    key=f"proficiency_{i}",
                    help="0=初学者, 50=中等, 100=专家"
                )
                st.session_state.skills_list[i]['proficiency'] = proficiency

            with col3:
                # 删除按钮
                if len(st.session_state.skills_list) > 1:
                    if st.button("🗑️", key=f"remove_{i}", help="删除这个技能"):
                        skills_to_remove.append(i)

            st.markdown('</div>', unsafe_allow_html=True)

    # 移除标记的技能
    for i in reversed(skills_to_remove):
        st.session_state.skills_list.pop(i)
        st.rerun()


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

        # 学历水平（改进点1）
        education_level = st.selectbox("学历水平", ["Bachelor", "Master", "PhD"])
        school_tier = st.selectbox(
            "学校层次",
            ["985/QS Top 50", "211/QS Top 200", "其他"],
            help="影响薪资预测的权重计算"
        )

        # 求职偏好
        st.subheader("求职偏好")

        # 经验级别（改进点3）
        experience_level = st.selectbox(
            "经验级别",
            ["入门级 (0-2年)", "中级 (2-5年)", "高级 (5-10年)", "专家级 (10年以上)"]
        )

        # 工作类型（改进点4）
        employment_type = st.selectbox(
            "工作类型",
            ["全职", "兼职", "合同工", "自由职业"]
        )

        company_location = st.selectbox(
            "公司位置",
            ["United States", "United Kingdom", "Canada", "Germany", "Netherlands", "China", "Singapore", "Other"]
        )

        # 公司规模（改进点5）
        company_size = st.selectbox(
            "公司规模",
            ["小型公司 (<50人)", "中型公司 (50-250人)", "大型公司 (>250人)"]
        )

        # 远程工作比例（改进点6）
        remote_ratio = st.selectbox(
            "远程工作比例",
            ["0% (完全现场办公)", "25% (偶尔远程)", "50% (混合办公)", "75% (主要远程)", "100% (完全远程)"]
        )

        # 行业（改进点7）
        industry = st.selectbox(
            "行业",
            ["Technology", "Finance", "Healthcare", "Media", "Retail", "Energy", "其他"]
        )

        # 分析按钮
        if st.button("🚀 开始分析", type="primary"):
            # 保存用户数据
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

    # 主页面：输入或结果展示
    if not st.session_state.analysis_complete:
        # 技能输入界面（改进点2）
        skill_input_component()

        # 欢迎页面其余内容
        st.markdown("""
        ## 🚀 开始你的AI职业分析

        填写左侧的基本信息和上方的技能背景，我们将为你提供：

        - 🎯 **职业类型识别** - 基于技能聚类分析
        - 💰 **薪资预测** - 机器学习模型预测
        - 📊 **技能画像** - 可视化你的技能分布
        - 🧠 **个性化建议** - AI驱动的职业建议
        - 📈 **成长路径** - 针对性的技能提升建议

        ### 📋 使用指南
        1. 在左侧输入你的基本信息
        2. 在上方配置你的技能背景和熟练程度
        3. 点击"开始分析"获得完整报告
        """)

    else:
        # 结果展示页面
        profile = st.session_state.user_profile
        skills_data = st.session_state.skills_list

        # 技能聚类分析
        cluster_type, recommended_skills = analyze_skills_cluster(skills_data)

        # 薪资预测
        predicted_salary, lower_bound, upper_bound = predict_salary(
            profile, skills_data, salary_model, label_encoders
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

            # 显示用户技能
            if skills_data and any(skill['skill'] for skill in skills_data):
                skills_df = pd.DataFrame([
                    {"技能": skill['skill'], "熟练程度": skill['proficiency']}
                    for skill in skills_data if skill['skill']
                ])

                fig = px.bar(
                    skills_df,
                    x="熟练程度",
                    y="技能",
                    orientation='h',
                    title="你的技能熟练程度",
                    color="熟练程度",
                    color_continuous_scale="viridis"
                )
                fig.update_layout(height=400)
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

        # GPT建议
        st.markdown('<div class="sub-header">🧠 AI职业建议</div>', unsafe_allow_html=True)

        advice = generate_career_advice(
            profile, cluster_type, (predicted_salary, lower_bound, upper_bound), skills_data
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
                skills_summary = ", ".join([f"{s['skill']}({s['proficiency']}%)" for s in skills_data if s['skill']])
                result_text = f"""
                AI Career Compass 分析报告

                职业类型: {cluster_type}
                预测薪资: ${predicted_salary:,.0f}
                薪资范围: ${lower_bound:,.0f} - ${upper_bound:,.0f}
                经验年限: {profile['years_experience']} 年
                学历: {profile['education_required']} ({profile.get('school_tier', '其他')})

                技能背景: {skills_summary}

                {advice}
                """
                st.code(result_text)

        with col2:
            if st.button("🔄 重新分析"):
                st.session_state.analysis_complete = False
                st.rerun()


if __name__ == "__main__":
    main()