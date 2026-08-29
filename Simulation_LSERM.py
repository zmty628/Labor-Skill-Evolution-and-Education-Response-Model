import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

np.random.seed(2026)

# ==================== 全局参数与技能集 ====================
CAREERS = ['High-skill innovative occupations', 'Skill-intensive operational occupations', 'Knowledge-processing occupations']
SKILLS = ['编程', '数据分析', 'AI工具使用', '设计创意', '手工操作', '沟通协调', '伦理判断', '项目管理']
NUM_SKILLS = len(SKILLS)

INIT_SKILL_WEIGHTS = {
    'High-skill innovative occupations': [0.9, 0.8, 0.3, 0.2, 0.1, 0.5, 0.3, 0.6],
    'Skill-intensive operational occupations': [0.1, 0.2, 0.1, 0.4, 0.9, 0.6, 0.2, 0.3],
    'Knowledge-processing occupations': [0.2, 0.1, 0.2, 0.9, 0.3, 0.7, 0.4, 0.2]
}

AI_RELEVANCE = [0.9, 0.8, 1.0, 0.4, 0.2, 0.3, 0.7, 0.5]
AUTOMATION_RISK = [0.3, 0.4, 0.1, 0.7, 0.8, 0.2, 0.1, 0.3]

# 添加技能描述
SKILL_DESCRIPTIONS = {
    '编程': 'Programming & Coding',
    '数据分析': 'Data Analysis',
    'AI工具使用': 'AI Tool Usage',
    '设计创意': 'Design Creativity',
    '手工操作': 'Manual Operation',
    '沟通协调': 'Communication & Coordination',
    '伦理判断': 'Ethical Judgment',
    '项目管理': 'Project Management'
}


# ==================== 核心类定义 ====================
class LaborMarket:
    def __init__(self, career):
        self.career = career
        self.time = 0
        self.jobs = 1000
        self.skill_weights = np.array(INIT_SKILL_WEIGHTS[career])
        self.history_jobs = [self.jobs]
        self.history_weights = [self.skill_weights.copy()]
        self.history_match_rate = []
        self.job_uncertainty = []  # 新增：岗位不确定性记录

    def update_jobs(self, base_growth_rate=0.02, ai_penetration=0.1, creation_coef=0.3):
        new_base = self.jobs * base_growth_rate
        obsolete = self.jobs * ai_penetration * np.mean(AUTOMATION_RISK) * 0.5
        created = obsolete * creation_coef * (1 - np.exp(-self.time / 5))

        # 添加随机扰动
        random_factor = np.random.normal(1, 0.05)  # 5%的随机波动
        uncertainty = np.std([new_base, obsolete, created]) / self.jobs if self.jobs > 0 else 0

        self.jobs = max(100, self.jobs + (new_base - obsolete + created) * random_factor)
        self.history_jobs.append(self.jobs)
        self.job_uncertainty.append(uncertainty)

    def update_skill_weights(self, ai_influence=0.05):
        delta = ai_influence * (np.array(AI_RELEVANCE) - np.array(AUTOMATION_RISK))
        if self.career == 'Knowledge-processing occupations':
            delta[3] = 0.1  # 设计创意保持重要性
            delta[2] = 0.2  # AI工具使用重要性缓慢上升
        elif self.career == 'High-skill innovative occupations':
            delta[2] = 0.4  # High-skill innovative occupations领域AI工具使用重要性上升更快
            delta[0] = 0.3  # 编程重要性上升

        self.skill_weights = np.clip(self.skill_weights + delta, 0.1, 1.0)
        self.skill_weights = self.skill_weights / np.sum(self.skill_weights) * NUM_SKILLS / 5
        self.history_weights.append(self.skill_weights.copy())

    def step(self, ai_penetration_factor=0.1):
        self.time += 1
        self.update_jobs(ai_penetration=ai_penetration_factor * (1 - np.exp(-self.time / 3)))
        self.update_skill_weights(ai_influence=0.03 + 0.02 * np.tanh(self.time / 10))


class EducationInstitution:
    def __init__(self, career, market):
        self.career = career
        self.market = market
        self.time = 0
        self.enrollment = 200
        self.curriculum_weights = np.array(INIT_SKILL_WEIGHTS[career])
        self.history_enrollment = [self.enrollment]
        self.history_curriculum = [self.curriculum_weights.copy()]
        self.history_graduates = []
        self.match_rates = []  # 新增：匹配率记录

    def produce_graduates(self):
        noise = np.random.normal(0, 0.1, NUM_SKILLS)
        grad_skills = np.clip(self.curriculum_weights + noise, 0, 1)
        return grad_skills

    def calculate_match_rate(self, graduates_skills):
        if len(graduates_skills) == 0:
            return 0.5
        avg_grad = np.mean(graduates_skills, axis=0)
        dot = np.dot(avg_grad, self.market.skill_weights)
        norm = np.linalg.norm(avg_grad) * np.linalg.norm(self.market.skill_weights)
        match = dot / norm if norm > 0 else 0
        return match

    def adjust_enrollment(self, match_rate, target_match=0.8, sensitivity=0.3):
        # 添加惯性因子，避免过度调整
        inertia = 0.7
        adjustment = sensitivity * (match_rate - target_match) * inertia
        self.enrollment = int(max(50, self.enrollment * (1 + adjustment)))
        self.history_enrollment.append(self.enrollment)

    def adjust_curriculum(self, market_weights, learning_rate=0.2):
        # 添加课程调整的动量
        momentum = 0.3
        adjustment = learning_rate * (market_weights - self.curriculum_weights)
        self.curriculum_weights = self.curriculum_weights + adjustment * (1 + momentum)
        self.curriculum_weights = np.clip(self.curriculum_weights, 0.1, 1.0)
        self.curriculum_weights = self.curriculum_weights / np.sum(self.curriculum_weights) * NUM_SKILLS / 5
        self.history_curriculum.append(self.curriculum_weights.copy())

    def step(self, graduates_skills):
        self.time += 1
        match_rate = self.calculate_match_rate(graduates_skills)
        self.match_rates.append(match_rate)
        self.market.history_match_rate.append(match_rate)
        self.adjust_enrollment(match_rate)
        self.adjust_curriculum(self.market.skill_weights)
        return match_rate


# ==================== 模拟主循环 ====================
def run_simulation(years=10):
    markets = {career: LaborMarket(career) for career in CAREERS}
    institutions = {career: EducationInstitution(career, markets[career]) for career in CAREERS}

    results = {
        'jobs': {c: [] for c in CAREERS},
        'enrollment': {c: [] for c in CAREERS},
        'match_rate': {c: [] for c in CAREERS},
        'skill_gap': {c: [] for c in CAREERS},
        'job_uncertainty': {c: [] for c in CAREERS},  # 新增：不确定性
        'trend': {c: {'jobs': None, 'enrollment': None} for c in CAREERS}  # 新增：趋势线
    }

    for year in range(years):
        year_graduates = {c: [] for c in CAREERS}
        for career in CAREERS:
            inst = institutions[career]
            for _ in range(inst.enrollment):
                grad = inst.produce_graduates()
                year_graduates[career].append(grad)

        for career in CAREERS:
            market = markets[career]
            ai_pen = 0.15 if career == 'High-skill innovative occupations' else 0.1 if career == 'Skill-intensive operational occupations' else 0.08
            market.step(ai_penetration_factor=ai_pen)

        for career in CAREERS:
            inst = institutions[career]
            match_rate = inst.step(year_graduates[career])
            cos_sim = np.dot(inst.curriculum_weights, markets[career].skill_weights)
            norm = np.linalg.norm(inst.curriculum_weights) * np.linalg.norm(markets[career].skill_weights)
            skill_gap = 1 - cos_sim / norm if norm > 0 else 1

            results['jobs'][career].append(markets[career].jobs)
            results['enrollment'][career].append(inst.enrollment)
            results['match_rate'][career].append(match_rate)
            results['skill_gap'][career].append(skill_gap)
            results['job_uncertainty'][career].append(
                markets[career].job_uncertainty[-1] if markets[career].job_uncertainty else 0)

    # 计算趋势线
    for career in CAREERS:
        years_array = np.arange(len(results['jobs'][career]))
        # 岗位趋势
        slope_jobs, intercept_jobs, _, _, _ = stats.linregress(years_array, results['jobs'][career])
        results['trend'][career]['jobs'] = (slope_jobs, intercept_jobs)
        # 招生趋势
        slope_enroll, intercept_enroll, _, _, _ = stats.linregress(years_array, results['enrollment'][career])
        results['trend'][career]['enrollment'] = (slope_enroll, intercept_enroll)

    return markets, institutions, results


# ==================== 运行模拟 ====================
markets, institutions, results = run_simulation(years=8)

# ==================== 可视化（标题在图表下方） ====================
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(3, 2, figsize=(10, 18))
colors = {'High-skill innovative occupations': '#2E86AB', 'Skill-intensive operational occupations': '#A23B72', 'Knowledge-processing occupations': '#F18F01'}

# 1. Job Market Size with Trend Lines and Confidence Intervals
ax = axes[0, 0]
years = np.arange(8)
for career in CAREERS:
    # 绘制主曲线
    ax.plot(years, results['jobs'][career], label=career, marker='o', linewidth=2.5,
            color=colors[career], markersize=6)

    # 添加趋势线
    slope, intercept = results['trend'][career]['jobs']
    trend_line = intercept + slope * years
    ax.plot(years, trend_line, '--', color=colors[career], alpha=0.6, linewidth=1.5)

    # 添加不确定性区域（模拟置信区间）
    uncertainty = np.array(results['job_uncertainty'][career])
    upper_bound = results['jobs'][career] + results['jobs'][career] * uncertainty * 2
    lower_bound = results['jobs'][career] - results['jobs'][career] * uncertainty * 2
    ax.fill_between(years, lower_bound, upper_bound, alpha=0.1, color=colors[career])

    # 添加增长率注释
    growth_rate = (results['jobs'][career][-1] - results['jobs'][career][0]) / results['jobs'][career][0] * 100
    ax.text(years[-1], results['jobs'][career][-1], f'{growth_rate:+.1f}%',
            fontsize=9, ha='left', va='bottom', color=colors[career])

ax.set_xlabel('Year', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Jobs', fontsize=11, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)
ax.text(0.5, -0.25, '(a) Job Market Size Evolution with Trend Analysis', transform=ax.transAxes,
        fontsize=12, fontweight='bold', ha='center', va='top')

# 2. Enrollment Adjustment with Adaptive Response
ax = axes[0, 1]
for career in CAREERS:
    ax.plot(years, results['enrollment'][career], label=career, marker='s', linewidth=2.5,
            color=colors[career], markersize=6)

    # 添加趋势线
    slope, intercept = results['trend'][career]['enrollment']
    trend_line = intercept + slope * years
    ax.plot(years, trend_line, '--', color=colors[career], alpha=0.6, linewidth=1.5)

    # 标记调整点
    changes = np.diff(results['enrollment'][career])
    significant_changes = np.where(np.abs(changes) > 20)[0]
    for idx in significant_changes:
        ax.scatter(years[idx + 1], results['enrollment'][career][idx + 1],
                   color='red', s=80, zorder=5, marker='*')

ax.set_xlabel('Year', fontsize=11, fontweight='bold')
ax.set_ylabel('Enrollment Size', fontsize=11, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)
ax.text(0.5, -0.25, '(b) Enrollment Adjustment with Adaptive Responses', transform=ax.transAxes,
        fontsize=12, fontweight='bold', ha='center', va='top')

# 3. Skill Match Rate with Target Zone
ax = axes[1, 0]
for career in CAREERS:
    ax.plot(years, results['match_rate'][career], label=career, marker='^', linewidth=2.5,
            color=colors[career], markersize=6)

    # 添加最后值标记
    ax.annotate(f'{results["match_rate"][career][-1]:.3f}',
                xy=(years[-1], results['match_rate'][career][-1]),
                xytext=(5, 0), textcoords='offset points',
                fontsize=9, ha='left', color=colors[career])

# 添加目标区域
target_zone = ax.axhspan(0.75, 0.85, alpha=0.1, color='green', label='Optimal Zone')
ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Target (0.8)')

ax.set_xlabel('Year', fontsize=11, fontweight='bold')
ax.set_ylabel('Skill Match Rate', fontsize=11, fontweight='bold')
ax.set_ylim(0.4, 1.0)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.3)
ax.text(0.5, -0.25, '(c) Skill Match Rate with Optimal Target Zone', transform=ax.transAxes,
        fontsize=12, fontweight='bold', ha='center', va='top')

# 4. Skill Gap Evolution with Convergence Analysis
ax = axes[1, 1]
for career in CAREERS:
    ax.plot(years, results['skill_gap'][career], label=career, marker='d', linewidth=2.5,
            color=colors[career], markersize=6)

    # 添加收敛趋势线（指数衰减拟合）
    try:
        # 指数拟合：gap = a * exp(-b*t) + c
        popt = np.polyfit(years, np.log(np.array(results['skill_gap'][career]) + 1e-10), 1)
        fitted = np.exp(popt[1]) * np.exp(popt[0] * years)
        ax.plot(years, fitted, ':', color=colors[career], alpha=0.5, linewidth=1.5)
    except:
        pass

ax.set_xlabel('Year', fontsize=11, fontweight='bold')
ax.set_ylabel('Skill Gap (1 - Cosine Similarity)', fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)
ax.text(0.5, -0.25, '(d) Skill Gap Evolution with Convergence Trends', transform=ax.transAxes,
        fontsize=12, fontweight='bold', ha='center', va='top')

# 5. Final Skill Weights Comparison with Detailed Labels
ax = axes[2, 0]
career = 'High-skill innovative occupations'
x = np.arange(NUM_SKILLS)
width = 0.35

market_final = markets[career].skill_weights
curriculum_final = institutions[career].curriculum_weights

bars1 = ax.bar(x - width / 2, market_final, width, label='Market Demand',
               alpha=0.85, color='#2E86AB', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width / 2, curriculum_final, width, label='Curriculum',
               alpha=0.85, color='#F18F01', edgecolor='black', linewidth=0.5)

# 添加技能名称和数值
for i, (mkt, cur) in enumerate(zip(market_final, curriculum_final)):
    # 市场权重数值
    ax.text(i - width / 2, mkt + 0.02, f'{mkt:.2f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold', color='#2E86AB')
    # 课程权重数值
    ax.text(i + width / 2, cur + 0.02, f'{cur:.2f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold', color='#F18F01')
    # 差异箭头
    if abs(mkt - cur) > 0.1:
        y_max = max(mkt, cur)
        y_min = min(mkt, cur)
        ax.annotate('', xy=(i, y_max), xytext=(i, y_min),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))

# 设置x轴标签为技能缩写
skill_labels = ['Prog', 'DA', 'AI', 'Design', 'Manual', 'Comm', 'Ethics', 'PM']
ax.set_xticks(x)
ax.set_xticklabels(skill_labels, fontsize=9, fontweight='bold')
ax.set_xlabel('Skills', fontsize=11, fontweight='bold')
ax.set_ylabel('Normalized Weight', fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.3, axis='y')
ax.text(0.5, -0.25, '(e) High-skill innovative occupations: Market vs Curriculum Skill Weights Comparison', transform=ax.transAxes,
        fontsize=12, fontweight='bold', ha='center', va='top')

# 6. Skill Evolution Heatmap with Annotations
ax = axes[2, 1]
career = 'High-skill innovative occupations'
history_weights = markets[career].history_weights[:8]
weights_matrix = np.array(history_weights).T

im = ax.imshow(weights_matrix, aspect='auto', cmap='RdYlBu_r', vmin=0.1, vmax=0.9)

# 添加数值标签
for i in range(weights_matrix.shape[0]):
    for j in range(weights_matrix.shape[1]):
        ax.text(j, i, f'{weights_matrix[i, j]:.2f}',
                ha='center', va='center', fontsize=7,
                color='white' if weights_matrix[i, j] > 0.5 else 'black',
                fontweight='bold')

# 标记变化最大的技能
max_change_idx = np.argmax(np.std(weights_matrix, axis=1))
ax.axhline(max_change_idx + 0.5, color='yellow', linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlabel('Year', fontsize=11, fontweight='bold')
ax.set_ylabel('Skill Index', fontsize=11, fontweight='bold')
ax.set_xticks(range(8))
ax.set_xticklabels([f'Y{i + 1}' for i in range(8)], fontsize=9)
ax.set_yticks(range(NUM_SKILLS))
ax.set_yticklabels([f'{i}:{skill_labels[i]}' for i in range(NUM_SKILLS)], fontsize=8)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Skill Importance', fontsize=10, fontweight='bold')
ax.text(0.5, -0.25, '(f) High-skill innovative occupations Career: Skill Importance Evolution Over Time', transform=ax.transAxes,
        fontsize=12, fontweight='bold', ha='center', va='top')

plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.2, hspace=0.35, wspace=0.3)

plt.show()