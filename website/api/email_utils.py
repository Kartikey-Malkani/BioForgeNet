import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def _send_via_formspree(
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
    """HTTP fallback: POST form data to Formspree (works even when SMTP is blocked)."""
    formspree_url = os.getenv("FORMSPREE_URL", "https://formspree.io/f/mojkwezo").strip()
    try:
        import httpx
        payload = {
            "firstName": first_name,
            "lastName": last_name,
            "email": email,
            "company": company_name,
            "phone": phone,
            "industry": industry,
            "teamSize": company_size,
            "useCase": use_case or "",
            "message": message or "",
            "_subject": f"New BioForgeNet Demo Request — {company_name}",
            "_replyto": email,
        }
        resp = httpx.post(
            formspree_url,
            json=payload,
            headers={"Accept": "application/json"},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("ok"):
                print(f"✅ Formspree fallback email sent for {email}")
                return True
        print(f"⚠️ Formspree fallback returned status {resp.status_code}: {resp.text}")
        return False
    except Exception as exc:
        print(f"❌ Formspree fallback failed: {exc}")
        return False


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
    gmail_email = os.getenv("GMAIL_EMAIL", "").strip()
    gmail_password = os.getenv("GMAIL_APP_PASSWORD", "").strip()
    recipient_email = os.getenv("ADMIN_EMAIL", "kartikeymalkanitablet@gmail.com").strip()

    if not gmail_email or not gmail_password:
        print("⚠️ Warning: GMAIL_EMAIL or GMAIL_APP_PASSWORD not set — trying Formspree fallback")
        return _send_via_formspree(
            company_name, first_name, last_name, email, phone,
            industry, company_size, use_case, message,
        )

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🔬 New BioForgeNet Demo Request — {company_name}"
        msg["From"] = gmail_email
        msg["To"] = recipient_email

        text_content = f"""
New Demo Request from BioForgeNet

Company: {company_name}
Name: {first_name} {last_name}
Email: {email}
Phone: {phone}
Industry: {industry}
Company Size: {company_size}

Use Case: {use_case or 'Not specified'}

Additional Message:
{message or 'No additional message'}

Submitted: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""

        use_case_html = use_case if use_case else '<em style="color: #999;">Not specified</em>'
        message_html = message if message else '<em style="color: #999;">No additional message</em>'

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

        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_email, gmail_password)
            server.sendmail(gmail_email, recipient_email, msg.as_string())

        print(f"✅ Email sent successfully to {recipient_email}")
        return True
    except Exception as exc:
        print(f"❌ Failed to send email via SMTP: {exc}")
        print("🔄 Trying Formspree HTTP fallback...")
        return _send_via_formspree(
            company_name, first_name, last_name, email, phone,
            industry, company_size, use_case, message,
        )
