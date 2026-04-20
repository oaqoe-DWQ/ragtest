"""
邮件发送模块 - QQ邮箱发送评估结果
"""
import smtplib
import os
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# QQ邮箱SMTP配置
SMTP_SERVER = "smtp.qq.com"
SMTP_PORT = 465  # SSL端口


def load_email_config():
    """从环境变量加载邮件配置"""
    from dotenv import load_dotenv
    load_dotenv()
    return {
        'email_enabled': os.getenv("EMAIL_ENABLED", "false").lower() == "true",
        'recipient_email': os.getenv("RECIPIENT_EMAIL", ""),
        'sender_email': os.getenv("SENDER_EMAIL", "993541347@qq.com"),
        'sender_auth_code': os.getenv("SENDER_AUTH_CODE", ""),
    }


def send_result_email(
    recipient_email: str,
    excel_file: str,
    stats_file: str = None,
    subject: str = None,
    body: str = None,
    sender_email: str = None,
    sender_auth_code: str = None
):
    """
    发送评估结果邮件

    参数:
        recipient_email: 收件人邮箱
        excel_file: Excel结果文件路径
        stats_file: 统计报告文件路径（可选）
        subject: 邮件主题（可选，使用默认主题）
        body: 邮件正文（可选，使用默认正文）
        sender_email: 发件人邮箱（可选）
        sender_auth_code: 邮箱授权码（可选）

    返回:
        bool: 发送是否成功
    """
    # 使用传入的参数或配置文件的值
    config = load_email_config()
    auth_code = sender_auth_code or config['sender_auth_code']
    sender = sender_email or config['sender_email']

    if not auth_code:
        logger.error("未配置QQ邮箱授权码，无法发送邮件！")
        logger.error("请在.env文件中配置SENDER_AUTH_CODE")
        return False

    if not os.path.exists(excel_file):
        logger.error(f"Excel文件不存在: {excel_file}")
        return False

    # 默认邮件主题
    if subject is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        subject = f"【评估结果】数据评估已完成 - {timestamp}"

    # 默认邮件正文
    if body is None:
        import datetime
        body = f"""
您好，

评估任务已完成，附件包含处理结果。

生成时间：{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

此邮件由自动程序发送。
"""

    try:
        # 创建邮件
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # 添加正文
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # 添加Excel附件
        with open(excel_file, 'rb') as f:
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(f.read())
            encoders.encode_base64(attachment)
            filename = os.path.basename(excel_file)
            attachment.add_header('Content-Disposition', f'attachment; filename="{filename}"')
            msg.attach(attachment)

        # 添加统计报告附件
        if stats_file and os.path.exists(stats_file):
            with open(stats_file, 'rb') as f:
                attachment = MIMEBase('application', 'octet-stream')
                attachment.set_payload(f.read())
                encoders.encode_base64(attachment)
                filename = os.path.basename(stats_file)
                attachment.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                msg.attach(attachment)

        # 发送邮件（使用SSL）
        logger.info(f"正在连接QQ邮箱SMTP服务器...")
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            logger.info(f"正在登录邮箱...")
            server.login(sender, auth_code)
            logger.info(f"正在发送邮件至 {recipient_email}...")
            server.send_message(msg)

        logger.info(f"✅ 邮件发送成功！")
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error("QQ邮箱授权码验证失败，请检查授权码是否正确")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP发送失败: {e}")
        return False
    except Exception as e:
        logger.error(f"邮件发送异常: {e}")
        return False


def send_evaluation_result_email(ragas_results: Dict[str, Any], detail_file: str, report_file: str) -> bool:
    """
    发送Ragas评估结果邮件（包含测试结果、测试数据集、测试详细结果）

    参数:
        ragas_results: Ragas评估结果字典
        detail_file: 详细结果Excel文件路径
        report_file: 总体测试报告Excel文件路径

    返回:
        bool: 发送是否成功
    """
    import datetime

    config = load_email_config()

    if not config['email_enabled']:
        logger.info("邮件发送功能未启用，跳过发送")
        return False

    if not config['recipient_email']:
        logger.error("未配置收件人邮箱，请设置RECIPIENT_EMAIL")
        return False

    if not config['sender_auth_code']:
        logger.error("未配置发件人授权码，请设置SENDER_AUTH_CODE")
        return False

    # 生成邮件主题
    dataset_file = ragas_results.get('dataset_file', '未知数据集')
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    subject = f"【RAG评估报告】{dataset_file} - {timestamp}"

    # 构建邮件正文
    fallback_mode = ragas_results.get('fallback_mode', False)
    error_message = ragas_results.get('error_message', '')

    # 提取各项指标
    metrics = [
        ("Faithfulness", "忠实度", "faithfulness"),
        ("Answer Relevancy", "回答相关性", "answer_relevancy"),
        ("Context Precision", "上下文精确度", "context_precision"),
        ("Context Recall", "上下文召回率", "context_recall"),
        ("Context Entity Recall", "上下文实体召回率", "context_entity_recall"),
        ("Context Relevance", "上下文相关性", "context_relevance"),
        ("Answer Correctness", "回答正确性", "answer_correctness"),
        ("Answer Similarity", "回答相似度", "answer_similarity"),
    ]

    # 生成指标表格
    metrics_table = "| 指标 | 中文名称 | 分数 | 百分比 |\n|------|----------|------|--------|\n"
    valid_scores = []
    for eng_name, cn_name, metric_key in metrics:
        value = ragas_results.get(metric_key)
        if value is not None:
            try:
                score = float(value)
                percentage = f"{score * 100:.1f}%"
                valid_scores.append(score)
            except (ValueError, TypeError):
                score = "N/A"
                percentage = "N/A"
        else:
            score = "N/A"
            percentage = "N/A"
        metrics_table += f"| {eng_name} | {cn_name} | {score} | {percentage} |\n"

    # 计算平均值
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        avg_percentage = f"{avg_score * 100:.1f}%"
    else:
        avg_score = "N/A"
        avg_percentage = "N/A"

    # 样本数量
    sample_data = ragas_results.get('sample_data', [])
    sample_count = len(sample_data) if sample_data else 0

    # 构建HTML邮件正文
    body_html = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
        .header {{ background-color: #4472C4; color: white; padding: 20px; text-align: center; }}
        .content {{ padding: 20px; }}
        .metrics-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .metrics-table th {{ background-color: #4472C4; color: white; padding: 10px; text-align: left; }}
        .metrics-table td {{ padding: 8px; border: 1px solid #ddd; }}
        .metrics-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .footer {{ background-color: #f5f5f5; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
        .summary {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        .warning {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 15px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RAG 评估结果报告</h1>
        <p>评估时间: {timestamp}</p>
    </div>
    <div class="content">
        <h2>评估概览</h2>
        <div class="summary">
            <p><strong>数据集文件:</strong> {dataset_file}</p>
            <p><strong>样本数量:</strong> {sample_count} 个</p>
            <p><strong>评估模式:</strong> {'Fallback模式（使用默认分数）' if fallback_mode else '正常模式'}</p>
            <p><strong>平均分数:</strong> {avg_score:.4f} ({avg_percentage})</p>
        </div>
"""

    if error_message:
        body_html += f"""
        <div class="warning">
            <p><strong>⚠️ 警告:</strong> {error_message}</p>
        </div>
"""

    body_html += f"""
        <h2>评估指标详情</h2>
        <table class="metrics-table">
            <tr>
                <th>指标名称</th>
                <th>中文名称</th>
                <th>分数</th>
                <th>百分比</th>
            </tr>
"""

    for eng_name, cn_name, metric_key in metrics:
        value = ragas_results.get(metric_key)
        if value is not None:
            try:
                score = float(value)
                percentage = f"{score * 100:.1f}%"
                score_str = f"{score:.4f}"
            except (ValueError, TypeError):
                score_str = "N/A"
                percentage = "N/A"
        else:
            score_str = "N/A"
            percentage = "N/A"
        body_html += f"""
            <tr>
                <td>{eng_name}</td>
                <td>{cn_name}</td>
                <td>{score_str}</td>
                <td>{percentage}</td>
            </tr>
"""

    body_html += """
        </table>

        <h2>附件说明</h2>
        <ul>
            <li><strong>测试详细结果.xlsx</strong> - 包含每个样本的各项评估指标详情</li>
            <li><strong>总体测试报告.xlsx</strong> - 包含评估指标汇总和统计分析</li>
        </ul>

        <p style="color: #666; font-size: 12px;">
            本邮件由 RAG 评估系统自动发送。如有疑问，请联系系统管理员。
        </p>
    </div>
    <div class="footer">
        <p>RAG 评估系统 | 评估完成后自动发送</p>
    </div>
</body>
</html>
"""

    # 纯文本版本
    body_text = f"""
RAG 评估结果报告
================

评估时间: {timestamp}

【评估概览】
数据集文件: {dataset_file}
样本数量: {sample_count} 个
评估模式: {'Fallback模式（使用默认分数）' if fallback_mode else '正常模式'}
平均分数: {avg_score:.4f} ({avg_percentage})

"""

    if error_message:
        body_text += f"【警告】{error_message}\n\n"

    body_text += """【评估指标详情】
指标名称              | 中文名称     | 分数      | 百分比
--------------------|------------|----------|--------
"""

    for eng_name, cn_name, metric_key in metrics:
        value = ragas_results.get(metric_key)
        if value is not None:
            try:
                score = float(value)
                percentage = f"{score * 100:.1f}%"
                score_str = f"{score:.4f}"
            except (ValueError, TypeError):
                score_str = "N/A"
                percentage = "N/A"
        else:
            score_str = "N/A"
            percentage = "N/A"
        body_text += f"{eng_name:<22}| {cn_name:<10}| {score_str:<10}| {percentage}\n"

    body_text += f"""
【附件说明】
1. 测试详细结果.xlsx - 包含每个样本的各项评估指标详情
2. 总体测试报告.xlsx - 包含评估指标汇总和统计分析

---
本邮件由 RAG 评估系统自动发送。
"""

    try:
        # 创建邮件
        msg = MIMEMultipart('alternative')
        msg['From'] = config['sender_email']
        msg['To'] = config['recipient_email']
        msg['Subject'] = subject

        # 添加纯文本和HTML两个版本
        msg.attach(MIMEText(body_text, 'plain', 'utf-8'))
        msg.attach(MIMEText(body_html, 'html', 'utf-8'))

        # 添加详细结果Excel附件
        if os.path.exists(detail_file):
            with open(detail_file, 'rb') as f:
                attachment = MIMEBase('application', 'octet-stream')
                attachment.set_payload(f.read())
                encoders.encode_base64(attachment)
                filename = os.path.basename(detail_file)
                attachment.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                msg.attach(attachment)

        # 添加总体测试报告Excel附件
        if os.path.exists(report_file):
            with open(report_file, 'rb') as f:
                attachment = MIMEBase('application', 'octet-stream')
                attachment.set_payload(f.read())
                encoders.encode_base64(attachment)
                filename = os.path.basename(report_file)
                attachment.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                msg.attach(attachment)

        # 发送邮件
        logger.info(f"正在连接QQ邮箱SMTP服务器...")
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            logger.info(f"正在登录邮箱...")
            server.login(config['sender_email'], config['sender_auth_code'])
            logger.info(f"正在发送邮件至 {config['recipient_email']}...")
            server.send_message(msg)

        logger.info(f"✅ 评估结果邮件发送成功！")
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error("❌ QQ邮箱授权码验证失败，请检查授权码是否正确")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"❌ SMTP发送失败: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 邮件发送异常: {e}")
        return False


def send_simple_notification(recipient_email: str, message: str):
    """
    发送简单通知邮件（不带附件）

    参数:
        recipient_email: 收件人邮箱
        message: 通知内容
    """
    config = load_email_config()

    if not config['sender_auth_code']:
        logger.error("未配置QQ邮箱授权码，无法发送邮件！")
        return False

    try:
        msg = MIMEMultipart()
        msg['From'] = config['sender_email']
        msg['To'] = recipient_email
        msg['Subject'] = "【评估任务通知】"
        msg.attach(MIMEText(message, 'plain', 'utf-8'))

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(config['sender_email'], config['sender_auth_code'])
            server.send_message(msg)

        logger.info(f"通知邮件发送成功！")
        return True

    except Exception as e:
        logger.error(f"通知邮件发送失败: {e}")
        return False
