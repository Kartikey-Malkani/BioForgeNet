import os
from datetime import datetime


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
        summary = (
            f"Company: {company_name}\n"
            f"Contact: {first_name} {last_name}\n"
            f"Email: {email}\n"
            f"Phone: {phone}\n"
            f"Industry: {industry}\n"
            f"Company Size: {company_size}\n"
            f"Use Case: {use_case or 'Not specified'}\n\n"
            f"Message:\n{message or 'None'}\n\n"
            f"Submitted: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        payload = {
            "Company": company_name,
            "Name": f"{first_name} {last_name}",
            "Email": email,
            "Phone": phone,
            "Industry": industry,
            "Company Size": company_size,
            "Use Case": use_case or "Not specified",
            "Message": message or "None",
            "_subject": f"New BioForgeNet Demo Request — {company_name}",
            "_replyto": email,
            "message": summary,
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
    return _send_via_formspree(
        company_name,
        first_name,
        last_name,
        email,
        phone,
        industry,
        company_size,
        use_case,
        message,
    )
