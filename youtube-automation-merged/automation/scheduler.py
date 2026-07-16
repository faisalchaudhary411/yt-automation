"""
Task Scheduler (Stage 3)
========================
Runs the recurring automation jobs inside the live Flask process (the Repl is
a Reserved VM that stays up, so an in-process BackgroundScheduler is the right
fit — no external cron needed).

Jobs (each gated by a config flag):
  - comment replies     every COMMENT_CHECK_INTERVAL_MINUTES minutes
  - analytics snapshot  daily (feeds /analytics and the Telegram digest)
  - trending refresh    daily (keeps /trending fresh)
  - auto daily video    OPT-IN (AUTO_DAILY_VIDEO): generates one video/day from
                        the top trending topic, through the SAME pipeline +
                        Telegram approval gate as manual runs — nothing ever
                        publishes without your tap.

apscheduler is imported lazily inside start_scheduler() so the rest of the app
still runs if the package hasn't been installed yet.
"""


def _job_comment_replies():
    from automation import comments
    print("[scheduler] Running comment reply pass…")
    try:
        summary = comments.run_once()
        print(f"[scheduler] Comments: {summary['replies_posted']} replies, "
              f"{summary['spam_skipped']} spam skipped, {len(summary['errors'])} errors")
        if summary["replies_posted"]:
            from telegram_notifier import send_message
            send_message(
                f"💬 Comment automation: replied to {summary['replies_posted']} "
                f"comment(s) across {summary['videos_checked']} video(s)."
            )
    except Exception as e:
        print(f"[scheduler] Comment pass failed: {e}")


def _job_analytics_snapshot():
    from automation import analytics
    print("[scheduler] Taking analytics snapshot…")
    try:
        analytics.collect_snapshot()
        from config import ANALYTICS_TELEGRAM_DIGEST
        if ANALYTICS_TELEGRAM_DIGEST:
            report = analytics.latest_report()
            ch = report.get("channel") or {}
            delta = report.get("channel_delta") or {}
            if ch:
                from telegram_notifier import send_message
                send_message(
                    "📊 Daily channel digest\n"
                    f"Subscribers: {ch.get('subscribers', 0):,} "
                    f"(+{delta.get('subscribers_delta', 0):,})\n"
                    f"Total views: {ch.get('total_views', 0):,} "
                    f"(+{delta.get('views_delta', 0):,})\n"
                    f"Videos tracked: {len(report.get('videos', []))}"
                )
    except Exception as e:
        print(f"[scheduler] Analytics snapshot failed: {e}")


def _job_trending_refresh():
    from automation import trending
    print("[scheduler] Refreshing trending topics…")
    try:
        topics = trending.refresh_trending()
        print(f"[scheduler] Trending refreshed: {len(topics)} topics")
    except Exception as e:
        print(f"[scheduler] Trending refresh failed: {e}")


def _job_daily_video():
    """Opt-in: generate one video from today's top trending topic, via the
    normal pipeline (still requires Telegram approval to publish)."""
    from automation import trending
    print("[scheduler] Auto daily video: picking a trending topic…")
    topics = trending.get_trending()
    if not topics:
        print("[scheduler] No trending topics available — skipping daily video.")
        return
    topic = topics[0]["title"]
    print(f"[scheduler] Auto daily video topic: {topic}")
    # Imported here to avoid a circular import at module load time.
    import main
    import uuid
    job_id = uuid.uuid4().hex[:12]
    with main.JOBS_LOCK:
        main.JOBS[job_id] = {"step": "queued", "topic": topic, "progress": 0.0,
                             "detail": "Queued (auto daily video)"}
    import threading
    threading.Thread(target=main.run_pipeline_job, args=(job_id, topic), daemon=True).start()


def start_scheduler():
    """Starts the background scheduler. Returns the scheduler (or None if
    disabled / apscheduler missing). Never raises — automation must never
    take the web app down with it."""
    from config import (
        SCHEDULER_ENABLED, AUTO_REPLY_ENABLED, COMMENT_CHECK_INTERVAL_MINUTES,
        AUTO_DAILY_VIDEO,
    )
    if not SCHEDULER_ENABLED:
        print("[scheduler] Disabled via config (SCHEDULER_ENABLED=False).")
        return None

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        print("[scheduler] apscheduler not installed — automation jobs OFF. "
              "Add 'apscheduler' to requirements.txt and reinstall.")
        return None

    scheduler = BackgroundScheduler(daemon=True)

    if AUTO_REPLY_ENABLED:
        scheduler.add_job(_job_comment_replies,
                          IntervalTrigger(minutes=COMMENT_CHECK_INTERVAL_MINUTES),
                          id="comment_replies", replace_existing=True,
                          max_instances=1, coalesce=True)

    scheduler.add_job(_job_analytics_snapshot,
                      CronTrigger(hour=8, minute=0),
                      id="analytics_daily", replace_existing=True,
                      max_instances=1, coalesce=True)

    scheduler.add_job(_job_trending_refresh,
                      CronTrigger(hour=7, minute=30),
                      id="trending_daily", replace_existing=True,
                      max_instances=1, coalesce=True)

    if AUTO_DAILY_VIDEO:
        scheduler.add_job(_job_daily_video,
                          CronTrigger(hour=9, minute=0),
                          id="daily_video", replace_existing=True,
                          max_instances=1, coalesce=True)

    scheduler.start()
    print(f"[scheduler] Started with jobs: {[j.id for j in scheduler.get_jobs()]}")
    return scheduler


def scheduler_status(scheduler) -> list:
    if not scheduler:
        return []
    return [{
        "id": job.id,
        "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
    } for job in scheduler.get_jobs()]
