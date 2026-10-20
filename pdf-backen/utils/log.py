import logging
import csv
import os
import glob
import time
from datetime import datetime

PURPLE = "\033[35m"  # 紫色
RESET = "\033[0m"    # 重置颜色
# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format=PURPLE+ '%(levelname)s'+ RESET +':     [%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M'  # 设置时间格式
)
logger = logging.getLogger('')

# ═══════════════════════════════════════════════════════════════════════
# File handler: 2026-10-13 新增
#   · 把所有 debug() 写一份到 logs/debug-YYYYMMDD.log (单文件, 1 天一切)。
#   · 启动时清掉比"今天"早一天以上的旧 log, 满足"保留一天即可"。
#   · basicConfig 只配了 stream handler, 我们补一个 TimedRotatingFileHandler。
#   · 路径: ./logs/ (与现有 csv 日志同级, 不污染)
# ═══════════════════════════════════════════════════════════════════════

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
_LOG_DIR = os.path.abspath(_LOG_DIR)


def _purge_old_debug_logs(keep_days: int = 1) -> None:
    """启动时清理: 保留今天 + 昨天(keep_days=1), 更早的全部删。

    配合 TimedRotatingFileHandler 的 when='midnight' + backupCount=1,
    运行时只保留 1 份(昨天的)滚动历史; 启动时再扫一次, 兜底手工残留。
    """
    if not os.path.isdir(_LOG_DIR):
        return
    cutoff = time.time() - keep_days * 86400
    purged = 0
    for path in glob.glob(os.path.join(_LOG_DIR, "debug-*.log*")):
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if mtime < cutoff:
            try:
                os.remove(path)
                purged += 1
            except OSError:
                # 文件可能被锁 / 权限不足, 不影响主流程
                pass
    if purged:
        # 这条只会打到 stderr 一次 (handler 还没装), 调试可观察
        print(f"[log] purged {purged} old debug log(s) (>{keep_days}d)")


def _install_file_handler() -> None:
    """给 root logger 装一个 daily rotating file handler, 与 stream 并存。

    设计要点:
      · 不在 basicConfig 里配 — 让 console 输出保持原样 (紫色 INFO 前缀)。
      · handler 自己的 level/format 独立, 不影响现有 console。
      · 路径 ./logs/debug.log, 午夜滚动为 debug.log.YYYY-MM-DD,
        backupCount=1 即"保留一天" (昨天的日志, 今天的当前文件)。
      · encoding=utf-8 兜住中文 msg。
      · 启动时调一次 _purge_old_debug_logs 删更早的残留 (兜底 backupCount)。
    """
    if not os.path.isdir(_LOG_DIR):
        try:
            os.makedirs(_LOG_DIR, exist_ok=True)
        except OSError:
            return  # 没权限建目录就静默跳过, 不阻塞后端启动

    log_path = os.path.join(_LOG_DIR, "debug.log")

    # 防止重复装 (reload / reimport 场景); root logger 拿 set 判断
    for h in logger.handlers:
        if getattr(h, "_crystal_debug_file", False):
            return

    try:
        from logging.handlers import TimedRotatingFileHandler
        fh = TimedRotatingFileHandler(
            filename=log_path,
            when="midnight",
            interval=1,
            backupCount=1,        # 只保留昨天的滚动
            encoding="utf-8",
            delay=True,           # 首次 write 时再 open, 避免空文件
            utc=False,            # 用本地时间切 (与 dt_str 一致)
        )
    except Exception as e:
        # handler 装失败不影响主流程, console 输出仍在
        print(f"[log] install file handler failed: {type(e).__name__}: {e}")
        return

    fh.suffix = "%Y-%m-%d"  # 滚动后的文件名: debug.log.2026-10-13
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        fmt='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
    ))
    fh._crystal_debug_file = True  # 标记, 防止重复
    logger.addHandler(fh)

    # 启动时清掉 >1 天的旧文件 (兜底 backupCount 在跨进程时不保证清理)
    _purge_old_debug_logs(keep_days=1)


_install_file_handler()


def debug(text):
    logger.info(text)

def log(admin, target, oldcontent, newcontent):
    now = datetime.now()
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
    # 打开 CSV 文件并写入数据
    with open(f'./logs/log-{now.strftime("%Y%m%d")}.csv', mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 如果需要在文件中写入标题行（可选）
        if f.tell() == 0:  # 检查文件是否为空
            writer.writerow(["Time", "AdminId","Target","History", "Update"])  # 写入标题行

        # 写入数据行
        writer.writerow([now.strftime("%H:%M:%S"), admin, target, oldcontent, newcontent])
