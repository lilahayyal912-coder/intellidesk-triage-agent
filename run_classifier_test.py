import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from services.severity_classifier import classify, ClassificationError

incidents = [
    {
        "label": "1. Company-wide auth failure",
        "service": "Auth-Service",
        "error_message": (
            "All employees are unable to log in. The authentication service "
            "is returning HTTP 500 errors for every login attempt across all regions."
        ),
    },
    {
        "label": "2. Misaligned button (cosmetic)",
        "service": "Frontend-UI",
        "error_message": (
            "One user reports that a button on the dashboard is slightly misaligned. "
            "No functional impact reported."
        ),
    },
    {
        "label": "3. Payment transactions failing",
        "service": "Payment-Gateway",
        "error_message": (
            "Payment transactions are failing for the majority of customers. "
            "Charge and refund API endpoints are returning errors."
        ),
    },
    {
        "label": "4. Slow database queries (app still up)",
        "service": "Database",
        "error_message": (
            "Database queries are taking approximately 8 seconds to complete. "
            "The application is still operational but noticeably degraded."
        ),
    },
]

SEP = "-" * 70

for inc in incidents:
    print(SEP)
    print(f"  {inc['label']}")
    print(SEP)
    try:
        r = classify(inc["service"], inc["error_message"])
        print(f"  Severity            : {r.severity}")
        print(f"  Reason              : {r.short_reason}")
        print(f"  Immediate action    : {r.immediate_action}")
        print(f"  Requires escalation : {r.requires_human_escalation}")
    except ClassificationError as exc:
        print(f"  [ERROR] {exc}")
    print()
