/**
 * Ragas 评估指标配置管理
 * 功能：
 * 1. 打开/关闭设置弹框
 * 2. 加载/保存指标配置
 * 3. 更新选中指标数量
 * 4. 与后端API交互
 */

// 全部可选的Ragas指标
const ALL_RAGAS_METRICS = [
    'context_recall',
    'context_precision',
    'context_entity_recall',
    'context_relevance',
    'faithfulness',
    'answer_relevancy',
    'answer_correctness',
    'answer_similarity'
];

// 默认选中的指标（初始化时启用的指标）
const DEFAULT_SELECTED_METRICS = [
    'context_recall',
    'context_precision',
    'context_entity_recall',
    'context_relevance'
];

// 必选指标（不可取消）
// === 改动：取消必选限制，改为可自由选择 ===
// const REQUIRED_METRICS = ['context_recall', 'context_precision'];  // 原必选配置
const REQUIRED_METRICS = [];  // 改动：置空，允许自由选择任意指标组合

// 将配置暴露到全局，供其他脚本使用
window.ALL_RAGAS_METRICS = ALL_RAGAS_METRICS;
window.DEFAULT_SELECTED_RAGAS_METRICS = DEFAULT_SELECTED_METRICS;
if (!Array.isArray(window.currentRagasMetrics) || window.currentRagasMetrics.length === 0) {
    window.currentRagasMetrics = [...DEFAULT_SELECTED_METRICS];
}

/**
 * 打开Ragas设置弹框
 */
function openRagasSettings() {
    const modal = document.getElementById('ragas-settings-modal');
    modal.style.display = 'block';
    
    // 加载当前配置
    loadRagasMetricsConfig();
    
    // 添加关闭弹框的点击事件
    window.onclick = function(event) {
        if (event.target === modal) {
            closeRagasSettings();
        }
    };
}

/**
 * 关闭Ragas设置弹框
 */
function closeRagasSettings() {
    const modal = document.getElementById('ragas-settings-modal');
    modal.style.display = 'none';
}

/**
 * 从本地存储或API加载Ragas指标配置
 */
async function loadRagasMetricsConfig() {
    try {
        // 先尝试从后端API获取
        const response = await fetch('/api/ragas/config');
        let selectedMetrics = [...DEFAULT_SELECTED_METRICS];
        
        if (response.ok) {
            const data = await response.json();
            if (data.success && data.data.enabled_metrics) {
                selectedMetrics = data.data.enabled_metrics;
            }
        } else {
            // API失败，尝试从本地存储读取
            const stored = localStorage.getItem('ragas_metrics_config');
            if (stored) {
                selectedMetrics = JSON.parse(stored);
            }
        }
        
        // 更新复选框状态
        updateCheckboxes(selectedMetrics);
        
        // 更新选中数量
        updateSelectedCount();
        
        // 更新全局当前指标
        window.currentRagasMetrics = [...selectedMetrics];
        if (typeof refreshRagasMetricPlaceholders === 'function') {
            refreshRagasMetricPlaceholders();
        }
        
    } catch (error) {
        console.error('加载Ragas配置失败:', error);
        // 使用默认配置
        updateCheckboxes(DEFAULT_SELECTED_METRICS);
        updateSelectedCount();
        window.currentRagasMetrics = [...DEFAULT_SELECTED_METRICS];
        if (typeof refreshRagasMetricPlaceholders === 'function') {
            refreshRagasMetricPlaceholders();
        }
    }
}

/**
 * 更新复选框状态
 * @param {Array} selectedMetrics - 选中的指标列表
 */
function updateCheckboxes(selectedMetrics) {
    ALL_RAGAS_METRICS.forEach(metric => {
        const checkbox = document.getElementById(`metric-${metric}`);
        if (checkbox) {
            checkbox.checked = selectedMetrics.includes(metric);
            
            // 必选项保持禁用状态（REQUIRED_METRICS 为空时此逻辑不生效）
            if (REQUIRED_METRICS.includes(metric)) {
                checkbox.disabled = true;
                checkbox.checked = true;
            }
        }
    });
}

/**
 * 保存Ragas设置
 */
async function saveRagasSettings() {
    // 获取所有选中的指标
    const selectedMetrics = getSelectedMetrics();
    
    // 验证必选项
    // === 改动：取消必选校验，改为可自由选择任意指标 ===
    // if (!selectedMetrics.includes('context_recall') || !selectedMetrics.includes('context_precision')) {
    //     showToast('Context Recall 和 Context Precision 是必选指标', 'error');
    //     return;
    // }
    
    // 至少选择一个指标
    if (selectedMetrics.length === 0) {
        showToast('请至少选择一个评估指标', 'error');
        return;
    }
    
    try {
        // 保存到后端
        const response = await fetch('/api/ragas/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                enabled_metrics: selectedMetrics
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            // 同时保存到本地存储作为备份
            localStorage.setItem('ragas_metrics_config', JSON.stringify(selectedMetrics));
            window.currentRagasMetrics = [...selectedMetrics];
            
            // 显示成功消息
            showToast(`配置已保存！已选择 ${selectedMetrics.length} 个指标`, 'success');
            
            // 关闭弹框
            closeRagasSettings();
            
            // 更新UI显示（可选：更新按钮文本显示当前指标数量）
            updateRagasButtonText(selectedMetrics.length);
            if (typeof refreshRagasMetricPlaceholders === 'function') {
                refreshRagasMetricPlaceholders();
            }
            
        } else {
            showToast('保存配置失败: ' + result.message, 'error');
        }
        
    } catch (error) {
        console.error('保存Ragas配置失败:', error);
        
        // 即使API失败，也保存到本地存储
        localStorage.setItem('ragas_metrics_config', JSON.stringify(selectedMetrics));
        window.currentRagasMetrics = [...selectedMetrics];
        showToast('配置已保存到本地（服务器连接失败）', 'warning');
        closeRagasSettings();
        if (typeof refreshRagasMetricPlaceholders === 'function') {
            refreshRagasMetricPlaceholders();
        }
    }
}

/**
 * 获取所有选中的指标
 * @returns {Array} 选中的指标列表
 */
function getSelectedMetrics() {
    const selectedMetrics = [];
    
    ALL_RAGAS_METRICS.forEach(metric => {
        const checkbox = document.getElementById(`metric-${metric}`);
        if (checkbox && checkbox.checked) {
            selectedMetrics.push(metric);
        }
    });
    
    return selectedMetrics;
}

/**
 * 更新选中指标数量显示
 */
function updateSelectedCount() {
    const selectedMetrics = getSelectedMetrics();
    const countElement = document.getElementById('selected-metrics-count');
    if (countElement) {
        countElement.textContent = selectedMetrics.length;
    }
}

/**
 * 更新Ragas评估按钮文本（可选功能）
 * @param {number} count - 选中的指标数量
 */
function updateRagasButtonText(count) {
    const btn = document.getElementById('ragas-evaluate-btn');
    if (btn && !btn.disabled) {
        const icon = '<i class="fas fa-play"></i>';
        btn.innerHTML = `${icon} 开始评估 (${count}个指标)`;
    }
}

/**
 * 初始化Ragas设置（页面加载时调用）
 */
function initializeRagasSettings() {
    // 为所有复选框添加变更事件监听
    ALL_RAGAS_METRICS.forEach(metric => {
        const checkbox = document.getElementById(`metric-${metric}`);
        if (checkbox) {
            checkbox.addEventListener('change', updateSelectedCount);
        }
    });
    
    // 加载保存的配置
    loadRagasMetricsConfig();
}

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeRagasSettings();
});

