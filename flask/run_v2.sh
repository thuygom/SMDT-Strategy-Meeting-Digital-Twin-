#!/bin/bash
# v2_app.py 관리 스크립트 (start/stop/restart/status)

# 경로 설정 (스크립트 기준 상대경로)
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$BASE_DIR/server"
APP_FILE="v2_app.py"
PYTHON="$BASE_DIR/bin/python"
LOG_FILE="$BASE_DIR/v2_app.log"

start() {
    PID=$(pgrep -f "$APP_FILE")
    if [ -n "$PID" ]; then
        echo "$APP_FILE is already running with PID $PID"
        exit 1
    fi
    echo "Starting $APP_FILE..."
    nohup "$PYTHON" "$APP_DIR/$APP_FILE" > "$LOG_FILE" 2>&1 &
    echo "$APP_FILE started with PID $!"
}

stop() {
    PID=$(pgrep -f "$APP_FILE")
    if [ -n "$PID" ]; then
        echo "Stopping $APP_FILE (PID $PID)..."
        kill -9 $PID
        echo "Stopped."
    else
        echo "$APP_FILE is not running."
    fi
}

status() {
    PID=$(pgrep -f "$APP_FILE")
    if [ -n "$PID" ]; then
        echo "$APP_FILE is running with PID $PID"
    else
        echo "$APP_FILE is not running"
    fi
}

restart() {
    stop
    sleep 1
    start
}

case "$1" in
    start) start ;;
    stop) stop ;;
    restart) restart ;;
    status) status ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac

