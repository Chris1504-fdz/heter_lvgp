#!/usr/bin/env python
"""
notify_progress.py -- watch the standard-LVGP sweep and email at completion milestones.

Polls results/ for completed .mat files and emails when the sweep first crosses each of
1%, 5%, 10%, 50%, 100%. Remembers which milestones were already sent (state file), so it is
safe to restart. Run it alongside the sweep (own tmux window / background).

Two delivery methods:
  --method sendmail  (default)  use the node's local MTA (/usr/sbin/sendmail). No credentials,
                                but delivery to Gmail may land in spam or be rejected -- send a
                                --test first and check your inbox AND spam folder.
  --method gmail                use smtp.gmail.com:587 with a Gmail *App Password* (reliable).
                                Set env GMAIL_USER + GMAIL_APP_PASSWORD (NOT your normal password;
                                create one at Google Account -> Security -> App passwords).

Examples:
  python notify_progress.py --test                       # send one test email now, exit
  python notify_progress.py                               # watch, email via local sendmail
  GMAIL_USER=you@gmail.com GMAIL_APP_PASSWORD=xxxx \
      python notify_progress.py --method gmail            # watch, email via Gmail SMTP
"""
import os, glob, time, json, ssl, socket, smtplib, argparse, subprocess
from email.message import EmailMessage

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
STATE = os.path.join(HERE, ".notify_state.json")
TOTAL = 270
THRESHOLDS = [1, 5, 10, 50, 100]
TO_DEFAULT = "christianalejandro15@gmail.com"


def count_done():
    return len(glob.glob(os.path.join(RESULTS, "**", "*.mat"), recursive=True))


def send_email(subject, body, to, method, gmail_user, gmail_pass):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["To"] = to
    msg.set_content(body)
    if method == "gmail":
        if not (gmail_user and gmail_pass):
            raise RuntimeError("set GMAIL_USER and GMAIL_APP_PASSWORD for --method gmail")
        msg["From"] = gmail_user
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
            s.starttls(context=ssl.create_default_context())
            s.login(gmail_user, gmail_pass)
            s.send_message(msg)
    else:  # local sendmail
        msg["From"] = f"LVGP sweep <{os.environ.get('USER','user')}@{socket.getfqdn()}>"
        p = subprocess.run(["/usr/sbin/sendmail", "-t", "-i"], input=msg.as_bytes())
        if p.returncode != 0:
            raise RuntimeError(f"sendmail exited {p.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sendmail", "gmail"], default="sendmail")
    ap.add_argument("--to", default=TO_DEFAULT)
    ap.add_argument("--total", type=int, default=TOTAL)
    ap.add_argument("--interval", type=int, default=60, help="poll seconds")
    ap.add_argument("--test", action="store_true", help="send one test email and exit")
    ap.add_argument("--reset", action="store_true", help="forget already-sent milestones")
    args = ap.parse_args()

    gmail_user = os.environ.get("GMAIL_USER")
    gmail_pass = os.environ.get("GMAIL_APP_PASSWORD")

    def _send(sub, body):
        send_email(sub, body, args.to, args.method, gmail_user, gmail_pass)

    if args.test:
        n = count_done()
        _send(f"[LVGP sweep] test ({n}/{args.total} done)",
              f"Notification test from {socket.getfqdn()} via {args.method}.\n"
              f"Sweep currently at {n}/{args.total} runs. If you got this, milestone emails will work.")
        print(f"test email sent to {args.to} via {args.method}")
        return

    if args.reset and os.path.exists(STATE):
        os.remove(STATE)
    sent = set(json.load(open(STATE))) if os.path.exists(STATE) else set()

    print(f"watching {RESULTS} (total={args.total}); milestones {THRESHOLDS}%; "
          f"emailing {args.to} via {args.method}; already sent {sorted(sent)}")
    while True:
        n = count_done()
        pct = 100.0 * n / args.total
        for t in THRESHOLDS:
            if t not in sent and pct >= t:
                try:
                    _send(f"[LVGP sweep] {t}% milestone -- {n}/{args.total} ({pct:.1f}%)",
                          f"The standard-LVGP sweep reached the {t}% milestone.\n\n"
                          f"  completed : {n}/{args.total} runs ({pct:.1f}%)\n"
                          f"  host      : {socket.getfqdn()}\n"
                          f"  dir       : {RESULTS}\n")
                    sent.add(t)
                    json.dump(sorted(sent), open(STATE, "w"))
                    print(f"[{time.strftime('%H:%M:%S')}] emailed {t}% ({n}/{args.total})")
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] EMAIL FAILED for {t}%: {e}")
        if 100 in sent or n >= args.total:
            print("done -- 100% milestone sent.")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
