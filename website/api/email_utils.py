import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


def send_demo_request_email(
    company_name: str,
    first_name: str,
    last_name: str,
    email: str,
    phone: str,
    industry: str,
    company_size: str,
    use_case: str = None,
    message: str = None,
) -> bool:
    """
    Send demo request notification to admin email via Gmail SMTP.
    
    For this to work:
    1. Go to https://myaccount.google.com/security
    2. Enable 2-Factor Authentication if not enabled
    3. Create App Password (Mail, Windows)
    4. Set env vars: GMAIL_EMAIL and GMAIL_APP_PASSWORD
    
    Returns: True if email sent successfully, False otherwise
    """
    
    gmail_email = os.getenv("GMAIL_EMAIL", "").strip()
    gmail_password = os.getenv("GMAIL_APP_PASSWORD", "").strip()
    recipient_email = os.getenv("ADMIN_EMAIL", "kartikeymalkanitablet@gmail.com").strip()
    
    if not gmail_email or not gmail_password:
        print("⚠️  Warning: GMAIL_EMAIL or GMAIL_APP_PASSWORD not set in environment")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🔬 New BioForgeNet Demo Request — {company_name}"
        msg["From"] = gmail_email
        msg["To"] = recipient_email
        
        # Plain text version
        text_content = f"""
New Demo Request from BioForgeNet

Company: {company_name}
Name: {first_name} {last_name}
Email: {email}
Phone: {phone}
Industry: {industry}
Company Size: {company_size}

Use Case: {use_case or "Not specified"}

Additional Message:
{message or "No additional message"}

Submitted: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""

  use_case_html = use_case if use_case else '<em style="color: #999;">Not specified</em>'
  message_html = message if message else '<em style="color: #999;">No additional message</em>'
        
        # HTML version
        html_content = f"""
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333;">
  <div style="max-width: 600px; margin: 0 auto; padding: 20px; background: #f9fafb; border-radius: 8px;">
    <h2 style="color: #1f2937; margin-bottom: 20px;">🔬 New Demo Request</h2>
    
    <table style="width: 100%; border-collapse: collapse;">
      <tr style="background: #fff; border-bottom: 1px solid #e5e7eb;">
        <td style="padding: 12px; font-weight: 600; width: 120px;">Company:</td>
        <td style="padding: 12px;">{company_name}</td>
      </tr>
      <tr style="background: #f9fafb; border-bottom: 1px solid #e5e7eb;">
        <td style="padding: 12px; font-weight: 600; width: 120px;">Contact:</td>
        <td style="padding: 12px;">{first_name} {last_name}</td>
      </tr>
      <tr style="background: #fff; border-bottom: 1px solid #e5e7eb;">
        <td style="padding: 12px; font-weight: 600; width: 120px;">Email:</td>
        <td style="padding: 12px;"><a href="mailto:{email}" style="color: #0066cc;">{email}</a></td>
      </tr>
      <tr style="background: #f9fafb; border-bottom: 1px solid #e5e7eb;">
        <td style="padding: 12px; font-weight: 600; width: 120px;">Phone:</td>
        <td style="padding: 12px;"><a href="tel:{phone}" style="color: #0066cc;">{phone}</a></td>
      </tr>
      <tr style="background: #fff; border-bottom: 1px solid #e5e7eb;">
        <td style="padding: 12px; font-weight: 600; width: 120px;">Industry:</td>
        <td style="padding: 12px;">{industry}</td>
      </tr>
      <tr style="background: #f9fafb; border-bottom: 1px solid #e5e7eb;">
        <td style="padding: 12px; font-weight: 600; width: 120px;">Company Size:</td>
        <td style="padding: 12px;">{company_size}</td>
      </tr>
      <tr style="background: #fff; border-bottom: 1px solid #e5e7eb;">
        <td style="padding: 12px; font-weight: 600; width: 120px;">Use Case:</td>
        <td style="padding: 12px;">{use_case_html}</td>
      </tr>
      <tr style="background: #f9fafb;">
        <td colspan="2" style="padding: 12px; font-weight: 600;">Message:</td>
      </tr>
      <tr style="background: #fff;">
        <td colspan="2" style="padding: 12px; border: 1px solid #e5e7eb; background: #f9fafb; border-radius: 4px; white-space: pre-wrap;">{message_html}</td>
      </tr>
    </table>
    
    <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #e5e7eb; font-size: 12px; color: #666;">
      <p style="margin: 0;">Submitted: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
      <p style="margin: 5px 0 0;">From: <a href="https://bioforgenet.live" style="color: #0066cc;">bioforgenet.live</a></p>
    </div>
  </div>
</body>
</html>
"""
        
        part1 = MIMEText(text_content, "plain")
        part2 = MIMEText(html_content, "html")
        
        msg.attach(part1)
        msg.attach(part2)
        
        # Send via Gmail SMTP
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_email, gmail_password)
            server.sendmail(gmail_email, recipient_email, msg.as_string())
        
        print(f"✅ Email sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False
