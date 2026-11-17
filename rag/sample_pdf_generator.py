"""
Sample PDF Generator Utility using ReportLab.

Creates valid, standard multi-page IT policy PDF documents for testing
the RAG pipeline and Knowledge Base uploads.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_sample_it_pdf(output_path: str):
    """
    Constructs a well-formed 2-page PDF document containing IT Support SOPs.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(out_file), pagesize=letter)
    width, height = letter

    # --- PAGE 1: VPN & Remote Access ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 60, "Acme Corp - Global IT Support Policy & SOP Guide")
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 80, "Document ID: SOP-IT-2026 | Effective Date: November 2026 | Version: 2.4")
    c.line(50, height - 90, width - 50, height - 90)

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 120, "1. Corporate VPN Setup and Remote Access")

    c.setFont("Helvetica", 10)
    lines_p1 = [
        "Remote employees requiring access to internal resources, git repositories, or staging environments",
        "must connect through the AcmeConnect VPN client.",
        "",
        "Step 1: Download the AcmeConnect VPN client from the internal portal: https://portal.acme-it.corp/vpn.",
        "Step 2: Launch the client and enter gateway address 'gw-west.acme-it.corp' or 'gw-east.acme-it.corp'.",
        "Step 3: Authenticate using your corporate Single Sign-On (SSO) credentials and Okta MFA push.",
        "Step 4: Once connected, verify internet routing by pinging internal gateway at 10.0.0.1.",
        "",
        "Important Operational Rules:",
        "- VPN sessions automatically disconnect after 8 hours of continuous connection.",
        "- Split-tunneling is strictly disabled. All workstation traffic routes through corporate firewalls.",
        "- If you encounter Error 403 (Certificate Expired), submit an IT Incident ticket under Category 'Network'."
    ]

    y = height - 145
    for line in lines_p1:
        if line.startswith("Important") or line.startswith("Step"):
            c.setFont("Helvetica-Bold", 10)
        else:
            c.setFont("Helvetica", 10)
        c.drawString(50, y, line)
        y -= 16

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 40, "Confidential - Internal IT Operations Manual | Page 1 of 2")
    c.showPage()  # Ends page 1 and moves to page 2

    # --- PAGE 2: Passwords & Hardware Policy ---
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 60, "2. Corporate Password & Account Lockout Policy")

    c.setFont("Helvetica", 10)
    lines_p2 = [
        "To maintain compliance with ISO 27001 and SOC 2 security standards, password complexity is enforced:",
        "- Passwords must be at least 14 characters in length.",
        "- Must contain at least one uppercase letter, one lowercase letter, one number, and one special symbol.",
        "- Passwords expire every 90 days. Reusing the last 10 passwords is prohibited.",
        "- Account lockout triggers automatically after 5 consecutive failed login attempts.",
        "- Self-service account unlock and password reset is accessible at https://reset.acme-it.corp.",
        "",
        "3. Hardware Provisioning & Refresh Cycles",
        "- Standard engineering laptop refresh cycle is 36 months (Apple MacBook Pro 14\" / 16\" or Dell XPS 15).",
        "- General business and administrative staff refresh cycle is 48 months.",
        "- Damaged or malfunctioning equipment must be physically presented to the Tech Bar or reported via ticket.",
        "- Lost or stolen equipment must be reported to IT Security (security@acme.corp) within 2 hours of discovery.",
        "- Peripherals (keyboards, mice, USB-C docks) can be ordered directly via the Service Request Catalog."
    ]

    y = height - 85
    for line in lines_p2:
        if line.startswith("3.") or line.startswith("To maintain"):
            c.setFont("Helvetica-Bold", 10)
        else:
            c.setFont("Helvetica", 10)
        c.drawString(50, y, line)
        y -= 16

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 40, "Confidential - Internal IT Operations Manual | Page 2 of 3")
    c.showPage()

    # --- PAGE 3: Email & Outlook Authentication Troubleshooting ---
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 60, "4. Corporate Email & Outlook Authentication Troubleshooting")

    c.setFont("Helvetica", 10)
    lines_p3 = [
        "If Microsoft Outlook reports 'Need Password', 'Disconnected', will not open, crashes, or has authentication errors:",
        "",
        "Step 1: Check Corporate SSO & Okta MFA Status",
        "- Open a web browser and verify you can access corporate SSO at https://sso.acme-it.corp.",
        "- Ensure Okta Verify MFA push notifications are accepted on your registered mobile device.",
        "",
        "Step 2: Clear Cached Exchange Tokens and Credentials",
        "- Windows OS: Open Control Panel > Credential Manager > Windows Credentials.",
        "  Remove all credentials starting with 'MicrosoftOffice16_Data' or 'adal_cache'.",
        "- macOS: Open Keychain Access > Search 'Exchange' or 'Microsoft Office' > Delete stale tokens.",
        "",
        "Step 3: Verify Webmail and Server Status",
        "- Access webmail at https://outlook.office.com to rule out mailbox account lockouts.",
        "- If Outlook is not opening or hangs on startup, launch Outlook in Safe Mode ('outlook.exe /safe').",
        "- If webmail functions but desktop Outlook fails, select File > Account Settings > Repair.",
        "",
        "Step 4: Escalation & Incident Ticket Logging",
        "- If authentication or startup errors continue for more than 15 minutes, log an IT Incident ticket",
        "  under Category 'Email' with Priority 'High'. An Exchange engineer will assist within 1 hour."
    ]

    y = height - 85
    for line in lines_p3:
        if line.startswith("Step") or line.startswith("4."):
            c.setFont("Helvetica-Bold", 10)
        else:
            c.setFont("Helvetica", 10)
        c.drawString(50, y, line)
        y -= 16

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 40, "Confidential - Internal IT Operations Manual | Page 3 of 3")
    c.showPage()

    c.save()


if __name__ == "__main__":
    out = "AI_USM_GenAI_Project/data/knowledge_base/acme_it_sop.pdf"
    create_sample_it_pdf(out)
    print(f"Sample PDF successfully generated at {out}")
