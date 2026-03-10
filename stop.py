import psutil
import os

def stop_server():
    stopped = False
    current_pid = os.getpid()
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline')
            pid = proc.info.get('pid')
            
            # 自分自身のプロセス(stop.py)は終了しないように除外
            if pid == current_pid:
                continue
                
            if cmdline:
                cmd_str = ' '.join(cmdline).lower()
                # Pythonプロセスであり、コマンドラインに 'server.py' を含んでいるか確認
                if 'python' in proc.info.get('name', '').lower() and 'server.py' in cmd_str:
                    print(f"MCPサーバーを停止しています... (PID: {pid})")
                    proc.terminate()
                    stopped = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
            
    if not stopped:
        print("実行中の MCPサーバー(server.py) は見つかりませんでした。")
        
if __name__ == "__main__":
    stop_server()
