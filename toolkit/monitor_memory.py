#!/usr/bin/env python3

import psutil
import time
import os
import subprocess
import json
from datetime import datetime

def get_memory_info():
    """获取详细的内存使用信息"""
    # 系统内存
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    # GPU内存
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(', ')
            gpu_used = int(gpu_info[0])
            gpu_total = int(gpu_info[1])
        else:
            gpu_used, gpu_total = 0, 0
    except:
        gpu_used, gpu_total = 0, 0
    
    # 进程信息
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
        try:
            pinfo = proc.info
            if pinfo['memory_info'].rss > 100 * 1024 * 1024:  # 只显示使用超过100MB的进程
                processes.append({
                    'pid': pinfo['pid'],
                    'name': pinfo['name'],
                    'memory_mb': pinfo['memory_info'].rss / 1024 / 1024,
                    'cpu_percent': pinfo['cpu_percent']
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # 按内存使用排序
    processes.sort(key=lambda x: x['memory_mb'], reverse=True)
    
    return {
        'timestamp': datetime.now().isoformat(),
        'system_memory': {
            'total_gb': memory.total / 1024**3,
            'used_gb': memory.used / 1024**3,
            'available_gb': memory.available / 1024**3,
            'percent': memory.percent
        },
        'swap_memory': {
            'total_gb': swap.total / 1024**3,
            'used_gb': swap.used / 1024**3,
            'percent': swap.percent
        },
        'gpu_memory': {
            'used_mb': gpu_used,
            'total_mb': gpu_total,
            'percent': (gpu_used / gpu_total * 100) if gpu_total > 0 else 0
        },
        'top_processes': processes[:10]  # 只保留前10个进程
    }

def print_memory_status(info):
    """打印内存状态"""
    print(f"\n{'='*60}")
    print(f"时间: {info['timestamp']}")
    print(f"{'='*60}")
    
    # 系统内存
    sys_mem = info['system_memory']
    print(f"系统内存: {sys_mem['used_gb']:.1f}GB / {sys_mem['total_gb']:.1f}GB ({sys_mem['percent']:.1f}%)")
    print(f"可用内存: {sys_mem['available_gb']:.1f}GB")
    
    # Swap
    swap_mem = info['swap_memory']
    if swap_mem['total_gb'] > 0:
        print(f"Swap: {swap_mem['used_gb']:.1f}GB / {swap_mem['total_gb']:.1f}GB ({swap_mem['percent']:.1f}%)")
    else:
        print("Swap: 未配置")
    
    # GPU内存
    gpu_mem = info['gpu_memory']
    if gpu_mem['total_mb'] > 0:
        print(f"GPU内存: {gpu_mem['used_mb']}MB / {gpu_mem['total_mb']}MB ({gpu_mem['percent']:.1f}%)")
    else:
        print("GPU: 未检测到")
    
    # 内存使用警告
    if sys_mem['percent'] > 80:
        print("⚠️  警告: 系统内存使用率超过80%")
    elif sys_mem['percent'] > 60:
        print("⚠️  注意: 系统内存使用率超过60%")
    
    # 顶级进程
    print(f"\n内存使用最多的进程:")
    print(f"{'PID':<8} {'进程名':<20} {'内存(MB)':<10} {'CPU%':<8}")
    print("-" * 50)
    for proc in info['top_processes'][:5]:
        print(f"{proc['pid']:<8} {proc['name'][:19]:<20} {proc['memory_mb']:<10.1f} {proc['cpu_percent']:<8.1f}")

def save_log(info, log_file):
    """保存日志到文件"""
    with open(log_file, 'a') as f:
        f.write(json.dumps(info) + '\n')

def analyze_memory_trend(log_file, hours=1):
    """分析内存使用趋势"""
    if not os.path.exists(log_file):
        return
    
    print(f"\n📊 最近{hours}小时内存使用趋势:")
    
    # 读取日志
    logs = []
    cutoff_time = datetime.now().timestamp() - hours * 3600
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    log_time = datetime.fromisoformat(data['timestamp']).timestamp()
                    if log_time > cutoff_time:
                        logs.append(data)
                except:
                    continue
    except:
        return
    
    if not logs:
        print("没有足够的历史数据")
        return
    
    # 计算统计信息
    memory_usage = [log['system_memory']['percent'] for log in logs]
    gpu_usage = [log['gpu_memory']['percent'] for log in logs if log['gpu_memory']['total_mb'] > 0]
    
    print(f"系统内存使用率: 平均 {sum(memory_usage)/len(memory_usage):.1f}%, "
          f"最高 {max(memory_usage):.1f}%, 最低 {min(memory_usage):.1f}%")
    
    if gpu_usage:
        print(f"GPU内存使用率: 平均 {sum(gpu_usage)/len(gpu_usage):.1f}%, "
              f"最高 {max(gpu_usage):.1f}%, 最低 {min(gpu_usage):.1f}%")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='系统内存监控工具')
    parser.add_argument('--interval', '-i', type=int, default=10, help='监控间隔(秒)')
    parser.add_argument('--log-file', '-l', default='memory_monitor.log', help='日志文件路径')
    parser.add_argument('--analysis', '-a', action='store_true', help='显示内存使用趋势分析')
    parser.add_argument('--once', action='store_true', help='只运行一次')
    
    args = parser.parse_args()
    
    print("🔍 系统内存监控工具启动")
    print(f"监控间隔: {args.interval}秒")
    print(f"日志文件: {args.log_file}")
    print("按 Ctrl+C 停止监控")
    
    try:
        while True:
            # 获取内存信息
            info = get_memory_info()
            
            # 显示状态
            print_memory_status(info)
            
            # 保存日志
            save_log(info, args.log_file)
            
            # 显示趋势分析
            if args.analysis:
                analyze_memory_trend(args.log_file)
            
            # 如果只运行一次，则退出
            if args.once:
                break
            
            # 等待下次监控
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")
    except Exception as e:
        print(f"\n错误: {e}")

if __name__ == "__main__":
    main()
