#!/usr/bin/env python3
"""
G4FAdmin - GPT4Free Provider and Model Management Tool

Command Line Interface
"""

import sys
import argparse
from pathlib import Path

from g4fadmin.admin import G4FAdmin


def list_providers(admin: G4FAdmin, working_only: bool = False):
    """List providers"""
    if working_only:
        providers = admin.get_working_providers()
        print(f"\n✅ Available Providers ({len(providers)}):\n")
    else:
        providers = admin.get_all_providers()
        print(f"\n📋 All Providers ({len(providers)}):\n")
    
    for i, p in enumerate(providers, 1):
        status = "✅" if p.working else "❌"
        features = []
        
        if p.supports_stream:
            features.append("streaming")
        if p.supports_message_history:
            features.append("history")
        if p.auth_required:
            features.append("auth required")
        if p.models:
            features.append(f"{len(p.models)} models")
        
        feature_str = ", ".join(features) if features else "basic"
        
        print(f"{i:3d}. {status} {p.name:30s} [{feature_str}]")
        
        # If models exist, show the first few
        if p.models and len(p.models) <= 5:
            print(f"      Models: {', '.join(p.models)}")
        elif p.models:
            print(f"      Models: {', '.join(p.models[:3])}... (total {len(p.models)}个)")


def list_models(admin: G4FAdmin):
    """List models"""
    models = admin.get_all_models()
    print(f"\n📦 All Models ({len(models)}):\n")
    
    # 按provider数量排序
    sorted_models = sorted(models, key=lambda m: len(m.providers), reverse=True)
    
    for i, m in enumerate(sorted_models, 1):
        provider_count = len(m.providers)
        print(f"{i:3d}. {m.name:40s} ({provider_count} providers)")
        
        # 显示provider列表（前5个）
        if provider_count <= 5:
            print(f"      {', '.join(m.providers)}")
        else:
            print(f"      {', '.join(m.providers[:5])}... (+{provider_count-5})")


def test_provider(admin: G4FAdmin, provider_name: str, model_name: str = None, prompt: str = "1+1=?"):
    """测试provider"""
    print(f"\n🧪 测试Provider: {provider_name}" + (f" (model={model_name})" if model_name else "") + "\n")
    
    # 先检查provider是否存在
    all_providers = admin.get_all_providers()
    provider_info = next((p for p in all_providers if p.name == provider_name), None)
    
    if not provider_info:
        print(f"❌ Provider '{provider_name}' 不存在")
        return
    
    print(f"Provider信息:")
    print(f"  代码标记状态: {'✅ 可用' if provider_info.working else '❌ 不可用'}")
    print(f"  streaming输出: {'✅' if provider_info.supports_stream else '❌'}")
    print(f"  消息history: {'✅' if provider_info.supports_message_history else '❌'}")
    print(f"  需要认证: {'是' if provider_info.auth_required else '否'}")
    print(f"  SupportsModels: {len(provider_info.models)} 个")
    
    if provider_info.models:
        if model_name and model_name not in provider_info.models:
            print(f"  ⚠️  警告: 该provider可能不Supports模型 '{model_name}'")
        print(f"    示例Models: {', '.join(provider_info.models[:5])}")
    
    print(f"\n正在真实测试...")
    print(f"  提示词: \"{prompt}\"")
    success, result, resp_time = admin.test_provider(
        provider_name, 
        model_name=model_name,
        test_prompt=prompt, 
        timeout=15,
        verbose=False
    )
    
    if success:
        print(f"✅ 测试Success!")
        print(f"  Response时间: {resp_time:.2f}秒")
        print(f"  Response内容: {result}")
    else:
        print(f"❌ 测试Failed!")
        print(f"  Error信息: {result}")


def find_providers_for_model(admin: G4FAdmin, model_name: str, test_providers: bool = False):
    """查找Supports指定模型的providers"""
    print(f"\n🔍 查找Supports '{model_name}'  Providers:\n")
    
    providers = admin.find_providers_for_model(model_name)
    
    if not providers:
        print(f"❌ 没有找到Supports '{model_name}' 的provider")
        return
    
    print(f"✅ 找到 {len(providers)} 个Supports的providers:")
    
    if test_providers:
        print(f"\n🧪 真实测试这些providers...\n")
        for i, p in enumerate(providers, 1):
            success, response, resp_time = admin.test_provider(
                p,
                model_name=model_name,
                test_prompt="1+1=?",
                timeout=15
            )
            status = "✅" if success else "❌"
            time_str = f"{resp_time:.2f}s" if resp_time else "N/A"
            result = response[:50] if response else ""
            print(f"  {i}. {status} {p:25s} [{time_str}] {result}")
    else:
        for i, p in enumerate(providers, 1):
            print(f"  {i}. {p}")
        print(f"\n💡 提示: 使用 --test-find 可以真实测试这些providers")


def show_cookie_providers(admin: G4FAdmin):
    """显示需要cookie的providers及其配置状态"""
    import os
    
    # 定义需要cookie的providers
    cookie_providers = {
        'Cerebras': {
            'url': 'https://inference.cerebras.ai/',
            'auth_type': 'browser_cookie3',
            'models': ['llama-3.3-70b', 'deepseek-r1', 'llama3.1-70b', 'deepseek-r1-distill-llama-70b', 'llama-3.1-8b'],
            'env_var': None,
            'extra_deps': []
        },
        'MetaAIAccount': {
            'url': 'https://www.meta.ai/',
            'auth_type': 'browser_cookie3',
            'models': ['meta-ai'],
            'env_var': None,
            'extra_deps': []
        },
        'WhiteRabbitNeo': {
            'url': 'https://www.whiterabbitneo.com/',
            'auth_type': 'browser_cookie3',
            'models': ['default'],
            'env_var': None,
            'extra_deps': []
        },
        'Grok': {
            'url': 'https://x.com/i/grok',
            'auth_type': 'nodriver',
            'models': ['grok-2', 'grok-3', 'grok-4', 'grok-latest', 'grok-3-mini', 'grok-3-reasoning'],
            'env_var': None,
            'extra_deps': ['nodriver', 'platformdirs']
        },
        'Pi': {
            'url': 'https://pi.ai/',
            'auth_type': 'nodriver',
            'models': ['pi'],
            'env_var': None,
            'extra_deps': ['nodriver', 'platformdirs']
        },
        'HailuoAI': {
            'url': 'https://hailuoai.com/',
            'auth_type': 'nodriver',
            'models': ['minimax'],
            'env_var': None,
            'extra_deps': ['nodriver', 'platformdirs']
        },
        'Claude': {
            'url': 'https://claude.ai/',
            'auth_type': 'env_variable',
            'models': ['claude'],
            'env_var': 'CLAUDE_COOKIE',
            'extra_deps': []
        }
    }
    
    print(f"\n{'='*80}")
    print("🍪 需要 Cookie 认证的 Providers")
    print(f"{'='*80}\n")
    
    # 检查依赖
    try:
        import browser_cookie3
        browser_cookie3_installed = True
    except ImportError:
        browser_cookie3_installed = False
    
    try:
        import nodriver
        nodriver_installed = True
    except ImportError:
        nodriver_installed = False
    
    print("📦 依赖状态:")
    print(f"   {'✅' if browser_cookie3_installed else '❌'} browser-cookie3 {'已安装' if browser_cookie3_installed else '未安装 (pip install browser-cookie3)'}")
    print(f"   {'✅' if nodriver_installed else '❌'} nodriver {'已安装' if nodriver_installed else '未安装 (pip install nodriver platformdirs)'}")
    print()
    
    for provider_name, info in cookie_providers.items():
        print(f"📌 {provider_name}")
        print(f"   网站: {info['url']}")
        print(f"   Models: {', '.join(info['models'][:3])}" + (f" (+{len(info['models'])-3}个)" if len(info['models']) > 3 else ""))
        
        # 检查依赖
        deps_ok = True
        if info['extra_deps']:
            missing_deps = []
            for dep in info['extra_deps']:
                try:
                    __import__(dep)
                except ImportError:
                    missing_deps.append(dep)
                    deps_ok = False
            
            if missing_deps:
                print(f"   依赖: ❌ 需要安装 {', '.join(missing_deps)}")
                print(f"   安装: pip install {' '.join(missing_deps)}")
        
        # 检查配置状态
        if info['auth_type'] == 'env_variable':
            env_var = info['env_var']
            if os.environ.get(env_var):
                print(f"   状态: ✅ 环境变量 {env_var} 已设置")
            else:
                print(f"   状态: ❌ 需要设置环境变量 {env_var}")
                print(f"   配置: export {env_var}=\"your_cookie_value\"")
        elif info['auth_type'] == 'nodriver':
            if nodriver_installed and deps_ok:
                print(f"   状态: ✅ nodriver 已安装（自动浏览器控制）")
            else:
                print(f"   状态: ❌ 需要安装 nodriver 和 platformdirs")
        else:  # browser_cookie3
            if browser_cookie3_installed:
                print(f"   状态: ✅ 自动从浏览器读取（需要先在浏览器登录）")
            else:
                print(f"   状态: ❌ 需要安装 browser-cookie3")
        
        # 测试命令示例
        model_example = info['models'][0] if info['models'][0] != 'default' else ''
        if model_example:
            print(f"   测试: g4fadmin --test {provider_name} --model {model_example}")
            print(f"   聊天: g4fadmin --chat \"hi\" --chat-provider {provider_name} --chat-model \"{model_example}\"")
        else:
            print(f"   测试: g4fadmin --test {provider_name}")
            print(f"   聊天: g4fadmin --chat \"hi\" --chat-provider {provider_name}")
        print()
    
    print(f"{'='*80}")
    print("⚠️  远程服务器注意: Cookie 必须在运行代码的机器上获取")
    print("   如使用 SSH 连接，本地浏览器 cookie 无法使用")
    print("   推荐: 使用手动 cookie 或选择不需要认证的 provider")
    print(f"{'='*80}")
    print("📖 详细配置指南: 查看 COOKIE_SETUP.md File")
    print(f"{'='*80}\n")


def export_info(admin: G4FAdmin, output_dir: str = "output"):
    """导出信息"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    provider_file = output_path / "g4f_providers.json"
    model_file = output_path / "g4f_models.json"
    
    print(f"\n💾 导出信息到JSONFile...\n")
    
    admin.export_providers(str(provider_file))
    print(f"✅ Providers信息: {provider_file}")
    
    admin.export_models(str(model_file))
    print(f"✅ Models信息: {model_file}")


def chat_once(admin: G4FAdmin, message: str, provider: str = None, model: str = None, stream: bool = False):
    """单次聊天"""
    print(f"\n💬 发送消息: {message}")
    
    if stream:
        print(f"🤖 回复: ", end='', flush=True)
        used_provider = None
        used_model = None
        
        try:
            for item in admin.chat(message, provider=provider, model=model, stream=True):
                # 最后一个yield是元组 ("success"|"error", provider, model)
                if isinstance(item, tuple):
                    status = item[0]
                    if status == "success":
                        used_provider = item[1]
                        used_model = item[2]
                    elif status == "error":
                        print(f"\n❌ Error: {item[1]}")
                        return False
                else:
                    # 普通chunk
                    print(item, end='', flush=True)
            
            print()  # 换行
            if used_provider:
                print(f"✅ 使用: {used_provider}/{used_model}")
                return True
            else:
                return False
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False
    else:
        success, response, used_provider, used_model = admin.chat(
            message, 
            provider=provider, 
            model=model,
            stream=False
        )
        
        if success:
            print(f"🤖 回复: {response}")
            print(f"✅ 使用: {used_provider}/{used_model}")
        else:
            print(f"❌ Error: {response}")
            return False
    
    return True


def chat_interactive(admin: G4FAdmin, provider: str = None, model: str = None, stream: bool = False):
    """交互式聊天"""
    print("\n" + "="*70)
    print("💬 G4FAdmin 交互式聊天")
    print("="*70)
    
    if provider:
        print(f"📍 Provider: {provider}")
    else:
        print(f"📍 Provider: 自动选择（基于 successful.json）")
    
    if model:
        print(f"🎯 Model: {model}")
    else:
        print(f"🎯 Model: 默认")
    
    print(f"🌊 streaming输出: {'是' if stream else '否'}")
    print(f"\n💡 输入 'exit' 或 'quit' 退出，'clear' 清空history\n")
    print("="*70 + "\n")
    
    history = []
    
    while True:
        try:
            user_input = input("👤 你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 再见！\n")
                break
            
            if user_input.lower() in ['clear', 'reset']:
                history = []
                print("✅ history已清空\n")
                continue
            
            # 发送消息
            if stream:
                print("🤖 助手: ", end='', flush=True)
                used_provider = None
                used_model = None
                
                try:
                    for item in admin.chat(
                        user_input, 
                        provider=provider, 
                        model=model, 
                        stream=True,
                        history=history
                    ):
                        # 最后一个yield是元组
                        if isinstance(item, tuple):
                            status = item[0]
                            if status == "success":
                                used_provider = item[1]
                                used_model = item[2]
                            elif status == "error":
                                print(f"\n❌ Error: {item[1]}\n")
                                continue
                        else:
                            # 普通chunk
                            print(item, end='', flush=True)
                    
                    print()  # 换行
                    if used_provider:
                        print(f"    └─ [{used_provider}/{used_model}]\n")
                        # 注意：streaming模式下我们没有完整Response文本，需要收集
                        # 这里简化处理，不添加到history
                    
                except Exception as e:
                    print(f"\n❌ Error: {e}\n")
                    continue
            else:
                success, response, used_provider, used_model = admin.chat(
                    user_input,
                    provider=provider,
                    model=model,
                    stream=False,
                    history=history
                )
                
                if success:
                    print(f"🤖 助手: {response}")
                    print(f"    └─ [{used_provider}/{used_model}]\n")
                    
                    # 添加到history
                    history.append({"role": "user", "content": user_input})
                    history.append({"role": "assistant", "content": response})
                else:
                    print(f"❌ Error: {response}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 再见！\n")
            break
        except EOFError:
            print("\n\n👋 再见！\n")
            break


def main():
    parser = argparse.ArgumentParser(
        description="检查GPT4FreeAvailable Providers和Models（Supports真实API测试）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                    # 显示摘要
  %(prog)s --real-test                        # 显示摘要并真实测试推荐providers
  %(prog)s --list-providers                   # 列出所有providers
  %(prog)s --list-providers --working-only    # 只列出可用的
  %(prog)s --cookie-providers                 # 显示需要cookie的providers及配置
  %(prog)s --test Perplexity                  # 测试Perplexity provider
  %(prog)s --test Perplexity --model llama-3.1-70b  # 测试特定provider+model组合
  %(prog)s --batch-test                       # 批量测试推荐providers
  %(prog)s --find gpt-4                       # 查找Supportsgpt-4的providers
  %(prog)s --probe                            # 探测所有可用的provider-model组合
  %(prog)s --export                           # 导出到JSONFile
  %(prog)s --chat                             # 进入交互式聊天模式
  %(prog)s --chat "你好"                      # 发送单条消息
  %(prog)s --chat "解释量子计算" --stream     # 使用streaming输出
  %(prog)s --chat --chat-provider DeepInfra   # 指定provider聊天
  %(prog)s --chat --chat-provider Cerebras --chat-model llama-3.3-70b  # 使用cookie认证的provider
        """
    )
    
    parser.add_argument(
        '--list-providers',
        action='store_true',
        help='列出所有providers'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='列出所有models'
    )
    
    parser.add_argument(
        '--cookie-providers',
        action='store_true',
        help='显示需要cookie认证的providers及配置状态'
    )
    
    parser.add_argument(
        '--working-only',
        action='store_true',
        help='只显示可用的providers（配合--list-providers使用）'
    )
    
    parser.add_argument(
        '--test',
        type=str,
        metavar='PROVIDER',
        help='真实测试指定provider（实际调用API）'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        metavar='MODEL',
        help='与--test配合使用，指定测试的模型'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        default='1+1=?',
        help='测试提示词（默认: "1+1=?"）'
    )
    
    parser.add_argument(
        '--batch-test',
        action='store_true',
        help='批量测试推荐的providers（真实API调用）'
    )
    
    parser.add_argument(
        '--find',
        type=str,
        metavar='MODEL',
        help='查找Supports指定模型的providers'
    )
    
    parser.add_argument(
        '--test-find',
        action='store_true',
        help='配合--find使用，真实测试找到的providers'
    )
    
    parser.add_argument(
        '--test-model',
        nargs=2,
        metavar=('PROVIDER', 'MODEL'),
        help='Test specified provider and model的连通性'
    )
    
    parser.add_argument(
        '--export',
        action='store_true',
        help='导出provider和model信息到JSONFile'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='导出File的输出目录（默认: output）'
    )
    
    parser.add_argument(
        '--real-test',
        action='store_true',
        help='在摘要中包含真实API测试（默认只显示代码属性）'
    )
    
    parser.add_argument(
        '--probe',
        action='store_true',
        help='探测所有可用的provider-model组合（实测）并保存结果'
    )
    
    parser.add_argument(
        '--probe-timeout',
        type=int,
        default=15,
        help='探测时每个测试的超时时间（秒，默认15）'
    )
    
    parser.add_argument(
        '--probe-workers',
        type=int,
        default=8,
        help='探测时的并发线程数（默认8）'
    )
    
    parser.add_argument(
        '--chat',
        type=str,
        nargs='?',
        const='__interactive__',
        metavar='MESSAGE',
        help='聊天模式：无参数进入交互式，带参数发送单条消息'
    )
    
    parser.add_argument(
        '--chat-provider',
        type=str,
        metavar='PROVIDER',
        help='聊天时指定provider（默认自动选择）'
    )
    
    parser.add_argument(
        '--chat-model',
        type=str,
        metavar='MODEL',
        help='聊天时指定model（默认使用provider默认）'
    )
    
    parser.add_argument(
        '--stream',
        action='store_true',
        help='使用streaming输出（配合--chat使用）'
    )
    
    args = parser.parse_args()
    
    # 创建G4FAdmin实例
    admin = G4FAdmin()
    
    # 根据参数执行相应操作
    if args.chat is not None:
        # 聊天模式
        if args.chat == '__interactive__':
            # 交互式聊天
            chat_interactive(
                admin, 
                provider=args.chat_provider,
                model=args.chat_model,
                stream=args.stream
            )
        else:
            # 单次聊天
            chat_once(
                admin,
                message=args.chat,
                provider=args.chat_provider,
                model=args.chat_model,
                stream=args.stream
            )
    elif args.list_providers:
        list_providers(admin, working_only=args.working_only)
    elif args.list_models:
        list_models(admin)
    elif args.cookie_providers:
        show_cookie_providers(admin)
    elif args.test:
        test_provider(admin, args.test, model_name=args.model, prompt=args.prompt)
    elif args.batch_test:
        print("\n开始批量测试推荐的providers（真实API调用）...")
        print("这可能需要一些时间...\n")
        
        recommended = admin.get_recommended_providers(10)
        provider_names = [p.name for p in recommended]
        
        results = admin.batch_test_providers(
            provider_names=provider_names,
            test_prompt=args.prompt,
            timeout=15
        )
        
        print(f"\n{'='*90}")
        print(f"Batch Test Results - total 测试{len(results)}个providers")
        print(f"{'='*90}")
        print(f"{'Provider':<25} {'状态':<10} {'Response时间':<12} {'Response摘要':<40}")
        print(f"{'-'*90}")
        
        success_count = 0
        for provider_name, (success, response, resp_time) in results.items():
            status = "✅ Success" if success else "❌ Failed"
            time_str = f"{resp_time:.2f}s" if resp_time else "N/A"
            resp_str = (response[:37] + "...") if (success and response and len(response) > 40) else (response or "")
            
            print(f"{provider_name:<25} {status:<10} {time_str:<12} {resp_str:<40}")
            
            if success:
                success_count += 1
        
        print(f"\n总结: {success_count}/{len(results)} 个providers测试Success")
        print(f"{'='*90}\n")
    elif args.find:
        find_providers_for_model(admin, args.find, test_providers=args.test_find)
    elif args.test_model:
        provider_name, model_name = args.test_model
        test_provider(admin, provider_name, model_name=model_name, prompt=args.prompt)
    elif args.probe:
        print("\n🔍 开始探测所有provider-model组合...")
        print("这将实际测试所有组合，可能需要较长时间...\n")
        
        output_file = Path(args.output_dir) / "probe_results.json"
        result = admin.probe_all_working_combinations(
            test_prompt=args.prompt,
            timeout=args.probe_timeout,
            max_workers=args.probe_workers,
            output_file=str(output_file)
        )
        
        print(f"\n{'='*70}")
        print("探测完成！")
        print(f"{'='*70}")
        print(f"📊 总Provider数: {result['total_providers']}")
        print(f"✅ 可用Provider数: {result['working_providers']}")
        print(f"📦 测试组合数: {result['total_combinations']}")
        print(f"✅ Success组合数: {result['successful_combinations']}")
        print(f"\n查看详细结果: {output_file}")
        print(f"{'='*70}\n")
    elif args.export:
        export_info(admin, args.output_dir)
    else:
        # 默认显示摘要
        admin.print_summary(include_real_test=args.real_test)


if __name__ == "__main__":
    main()
