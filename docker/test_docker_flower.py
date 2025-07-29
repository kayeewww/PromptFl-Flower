#!/usr/bin/env python3
"""
测试Docker版本的PromptFL with Flower
"""

import subprocess
import time
import sys
import os

def run_command(cmd, timeout=300):
    """运行命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"

def test_docker_setup():
    """测试Docker环境"""
    print("🔍 Testing Docker setup...")
    
    # 检查Docker
    success, stdout, stderr = run_command("docker --version")
    if not success:
        print("❌ Docker not available")
        return False
    print(f"   ✅ Docker: {stdout.strip()}")
    
    # 检查Docker Compose
    success, stdout, stderr = run_command("docker-compose --version")
    if not success:
        print("❌ Docker Compose not available")
        return False
    print(f"   ✅ Docker Compose: {stdout.strip()}")
    
    return True

def test_build():
    """测试构建"""
    print("🔨 Testing Docker build...")
    
    # 确保在正确的目录中（当前已经在docker目录）
    success, stdout, stderr = run_command("docker-compose -f docker-compose-flower.yml build", timeout=600)
    if not success:
        print("❌ Docker build failed")
        print("STDERR:", stderr[-500:])
        return False
    
    print("   ✅ Docker build successful")
    return True

def test_quick_run():
    """测试快速运行"""
    print("🚀 Testing quick run...")
    
    # 确保在正确的目录中
    current_dir = os.getcwd()
    print(f"   📁 Current directory: {current_dir}")
    
    # 检查文件是否存在
    compose_file = "docker-compose-flower.yml"
    if not os.path.exists(compose_file):
        print(f"❌ Compose file not found: {compose_file}")
        print(f"   Files in current directory: {os.listdir('.')}")
        return False
    
    # 启动服务
    success, stdout, stderr = run_command("docker-compose -f docker-compose-flower.yml up -d")
    if not success:
        print("❌ Failed to start services")
        print("STDERR:", stderr[-500:])
        return False
    
    print("   ✅ Services started")
    
    # 等待服务启动
    print("   ⏳ Waiting for services to initialize...")
    time.sleep(30)
    
    # 检查服务状态
    success, stdout, stderr = run_command("docker-compose -f docker-compose-flower.yml ps")
    print("   📊 Service status:")
    print(stdout)
    
    # 检查日志
    print("   📋 Server logs (last 10 lines):")
    success, stdout, stderr = run_command("docker logs --tail 10 promptfl-flower-server")
    print(stdout)
    
    print("   📋 Client1 logs (last 10 lines):")
    success, stdout, stderr = run_command("docker logs --tail 10 promptfl-flower-client1")
    print(stdout)
    
    # 停止服务
    print("   🛑 Stopping services...")
    run_command("docker-compose -f docker-compose-flower.yml down")
    
    return True

def main():
    """主测试函数"""
    print("🌸 PromptFL with Flower - Docker Test")
    print("=" * 50)
    
    # 测试Docker环境
    if not test_docker_setup():
        return False
    
    # 测试构建
    if not test_build():
        return False
    
    # 测试运行
    if not test_quick_run():
        return False
    
    print("\n🎉 All Docker tests passed!")
    print("\n🚀 You can now run:")
    print("   cd docker")
    print("   bash run_flower.sh PromptFL cifar10 ViT-B/16 16 5 3 16")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)