"""
G4FAdmin - GPT4Free Provider Management Tool

Core Classes:
  - G4FAdmin: Main management class
  - ProviderInfo: Provider information dataclass  
  - ModelInfo: Model information dataclass
  - TestResult: Test result dataclass
  - AuthType: Authentication type enum

Main Features:
  1. Provider management: scan, filter, recommend providers
  2. Model management: get all models and their supporting providers
  3. Authentication detection: identify auth methods through testing
  4. Real testing: test provider/model combination availability
  5. Batch testing: concurrent testing of multiple combinations
  6. Data export: export to JSON format

Usage Example:
    >>> admin = G4FAdmin()
    >>> providers = admin.get_recommended_providers(5)
    >>> success, resp, time = admin.test_provider("ApiAirforce", "gpt-4")
"""

import logging
import time
import json
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import concurrent.futures

logger = logging.getLogger(__name__)


class AuthType(Enum):
    """Authentication type"""
    NONE = "none"
    API_KEY = "api_key"
    COOKIE = "cookie"
    TOKEN = "token"
    HAR_FILE = "har_file"
    ACCOUNT = "account"
    UNKNOWN = "unknown"


@dataclass
class ProviderInfo:
    """Provider information"""
    name: str
    working: bool
    supports_stream: bool
    supports_message_history: bool
    supports_system_message: bool
    models: List[str]
    auth_type: AuthType = AuthType.NONE
    auth_required: bool = False
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'working': self.working,
            'supports_stream': self.supports_stream,
            'supports_message_history': self.supports_message_history,
            'supports_system_message': self.supports_system_message,
            'models': self.models,
            'auth_type': self.auth_type.value,
            'auth_required': self.auth_required,
        }


@dataclass
class ModelInfo:
    """Model information"""
    name: str
    providers: List[str]
    
    def to_dict(self) -> dict:
        return {'name': self.name, 'providers': self.providers}


@dataclass
class TestResult:
    """Test result"""
    provider: str
    model: str
    success: bool
    response: Optional[str] = None
    response_time: Optional[float] = None
    auth_type: AuthType = AuthType.NONE
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            'provider': self.provider,
            'model': self.model,
            'success': self.success,
            'response': self.response,
            'response_time': self.response_time,
            'auth_type': self.auth_type.value,
            'timestamp': self.timestamp.isoformat(),
        }


class G4FAdmin:
    """G4F Provider管理工具"""
    
    # 默认黑名单
    DEFAULT_BLACKLIST = {
        "Copilot", "CopilotAccount", "OpenaiAccount", "OpenaiChat",
        "GithubCopilot", "LMArena", "Gemini", "AnyProvider"
    }
    
    # 认证关键词
    AUTH_KEYWORDS = {
        AuthType.API_KEY: ["api key", "api_key", "apikey"],
        AuthType.COOKIE: ["cookie", "__secure", "session"],
        AuthType.TOKEN: ["token", "bearer", "authorization"],
        AuthType.HAR_FILE: [".har", "har file", "browser_cookie"],
        AuthType.ACCOUNT: ["login", "account", "credentials"],
    }
    
    def __init__(self, blacklist: Optional[Set[str]] = None):
        """初始化"""
        try:
            import g4f
            self.g4f = g4f
        except ImportError:
            raise ImportError("请安装g4f: pip install -U g4f")
        
        self._blacklist = blacklist or self.DEFAULT_BLACKLIST
        self._providers_cache: Optional[List[ProviderInfo]] = None
        self._models_cache: Optional[List[ModelInfo]] = None
        self._last_scan_time: Optional[datetime] = None
    
    def get_all_providers(self, force_refresh: bool = False) -> List[ProviderInfo]:
        """获取所有provider"""
        if self._providers_cache and not force_refresh:
            return self._providers_cache
        
        providers = []
        for provider in self.g4f.Provider.__providers__:
            try:
                name = getattr(provider, '__name__', str(provider))
                if name in self._blacklist:
                    continue
                
                models = self._extract_models(provider)
                needs_auth = getattr(provider, 'needs_auth', False)
                
                info = ProviderInfo(
                    name=name,
                    working=getattr(provider, 'working', False),
                    supports_stream=getattr(provider, 'supports_stream', False),
                    supports_message_history=getattr(provider, 'supports_message_history', True),
                    supports_system_message=getattr(provider, 'supports_system_message', True),
                    models=models,
                    auth_type=AuthType.UNKNOWN if needs_auth else AuthType.NONE,
                    auth_required=needs_auth,
                )
                providers.append(info)
            except Exception as e:
                logger.warning(f"无法获取provider {provider}: {e}")
        
        self._providers_cache = providers
        self._last_scan_time = datetime.now()
        return providers
    
    def _extract_models(self, provider) -> List[str]:
        """提取文本模型列表"""
        models = []
        
        # 1. 首先检查models属性
        if hasattr(provider, 'models'):
            attr = provider.models
            if isinstance(attr, list):
                models = [m for m in attr if isinstance(m, str)]
            elif isinstance(attr, dict):
                models = [k for k in attr.keys() if isinstance(k, str)]
        
        # 2. 检查model_aliases，提取别名（keys）和真实模型名（values）
        if hasattr(provider, 'model_aliases'):
            aliases = provider.model_aliases
            if isinstance(aliases, dict):
                # 提取别名（用户可以使用的简短名称）
                alias_keys = [k for k in aliases.keys() if isinstance(k, str)]
                
                # 提取真实模型名
                alias_values = []
                for value in aliases.values():
                    if isinstance(value, str):
                        alias_values.append(value)
                    elif isinstance(value, list):
                        alias_values.extend([v for v in value if isinstance(v, str)])
                
                # 如果原本没有models，使用别名（更简短易用）
                if not models:
                    models = alias_keys
                # 否则补充别名到现有models
                else:
                    models.extend(alias_keys)
                    models = list(set(models))  # 去重
        
        # 3. 如果还是没有，检查default_model
        if not models and hasattr(provider, 'default_model'):
            default = provider.default_model
            if default and isinstance(default, str):
                models = [default]
        
        return models
    
    def get_working_providers(
        self, 
        require_stream: bool = False,
        require_auth: Optional[bool] = None
    ) -> List[ProviderInfo]:
        """获取可用provider"""
        providers = [p for p in self.get_all_providers() if p.working]
        if require_stream:
            providers = [p for p in providers if p.supports_stream]
        if require_auth is not None:
            providers = [p for p in providers if p.auth_required == require_auth]
        return providers
    
    def get_recommended_providers(self, top_n: int = 5) -> List[ProviderInfo]:
        """推荐provider (优先已知稳定的、无认证、支持历史、流式、模型多)"""
        candidates = self.get_working_providers(require_auth=False)
        
        # 过滤掉可能是音频/图像模型的provider
        audio_image_keywords = ['audio', 'fm', 'image', 'flux', 'dalle', 'midjourney', 'blackforest']
        candidates = [
            p for p in candidates 
            if not any(kw in p.name.lower() for kw in audio_image_keywords)
        ]
        
        # 加载已知稳定providers
        try:
            from .config import KNOWN_STABLE_PROVIDERS
            known_stable = set(KNOWN_STABLE_PROVIDERS)
        except ImportError:
            known_stable = set()
        
        def score(p: ProviderInfo) -> int:
            # 优先级：已知稳定 > 支持历史 > 流式 > 系统消息 > 模型数量
            s = 0
            if p.name in known_stable:
                s += 1000  # 已知稳定的优先级最高
            if p.supports_message_history:
                s += 100
            if p.supports_stream:
                s += 50
            if p.supports_system_message:
                s += 20
            s += min(len(p.models), 50)  # 模型数量，但不超过50
            return s
        
        return sorted(candidates, key=score, reverse=True)[:top_n]
    
    def get_all_models(self, force_refresh: bool = False) -> List[ModelInfo]:
        """获取所有模型"""
        if self._models_cache and not force_refresh:
            return self._models_cache
        
        model_providers: Dict[str, Set[str]] = {}
        for p in self.get_all_providers(force_refresh):
            if not p.working:
                continue
            for model in p.models:
                model_providers.setdefault(model, set()).add(p.name)
        
        models = [
            ModelInfo(name=m, providers=sorted(ps))
            for m, ps in sorted(model_providers.items())
        ]
        self._models_cache = models
        return models
    
    def find_providers_for_model(self, model_name: str) -> List[str]:
        """查找支持指定模型的provider"""
        model_lower = model_name.lower()
        providers = []
        for p in self.get_all_providers():
            if not p.working:
                continue
            for m in p.models:
                if model_lower in m.lower() or m.lower() in model_lower:
                    providers.append(p.name)
                    break
        return providers
    
    def _identify_auth_from_error(self, error_msg: str) -> AuthType:
        """从错误信息识别Authentication type"""
        error_lower = error_msg.lower()
        for auth_type, keywords in self.AUTH_KEYWORDS.items():
            if any(kw in error_lower for kw in keywords):
                return auth_type
        if any(w in error_lower for w in ["auth", "unauthor", "forbidden", "403"]):
            return AuthType.UNKNOWN
        return AuthType.NONE
    
    def _extract_text_from_response(self, response) -> str:
        """提取响应文本"""
        try:
            from json_repair import repair_json
            has_repair = True
        except ImportError:
            has_repair = False
        
        # 字符串
        if isinstance(response, str):
            if has_repair:
                try:
                    response = json.loads(repair_json(response))
                except:
                    return response.strip()
            else:
                return response.strip()
        
        # 字典
        if isinstance(response, dict):
            # 错误响应
            if 'error_code' in response or response.get('status') == 'failed':
                return f"[错误] {response.get('text', response.get('error_code', '未知'))}"
            
            # 标准格式
            if 'choices' in response and response['choices']:
                choice = response['choices'][0]
                if isinstance(choice, dict):
                    if 'message' in choice:
                        return choice['message'].get('content', '').strip()
                    if 'text' in choice:
                        return choice['text'].strip()
            
            # 其他格式
            for key in ['text', 'content', 'answer', 'message']:
                if key in response and response[key]:
                    return str(response[key]).strip()
            
            return f"[未解析] {str(response)[:100]}..."
        
        # 生成器
        if hasattr(response, '__iter__') and not isinstance(response, (str, dict)):
            try:
                chunks = [str(c).strip() for c in response if str(c).strip() and str(c) != '{}']
                return ''.join(chunks).strip()
            except:
                pass
        
        return str(response).strip()
    
    def test_provider(
        self,
        provider_name: str,
        model_name: Optional[str] = None,
        test_prompt: str = "Hello",
        timeout: int = 15,
        verbose: bool = False
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """测试provider"""
        try:
            if not hasattr(self.g4f.Provider, provider_name):
                return False, f"Provider {provider_name} 不存在", None
            
            provider = getattr(self.g4f.Provider, provider_name)
            
            # BlackboxPro特殊处理：只接受空字符串作为model
            # 它的models列表包含很多模型名，但实际API只接受空字符串
            if provider_name == 'BlackboxPro':
                model_name = ''  # 强制使用空字符串
            
            # 自动选择模型
            if model_name is None:
                if hasattr(provider, 'default_model'):
                    model_name = provider.default_model
                elif hasattr(provider, 'models') and provider.models:
                    models = provider.models
                    if isinstance(models, list):
                        # 过滤空字符串
                        valid_models = [m for m in models if m]
                        model_name = valid_models[0] if valid_models else 'auto'
                    else:
                        model_name = 'auto'
                else:
                    model_name = 'auto'
            
            start_time = time.time()
            response = self.g4f.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": test_prompt}],
                provider=provider,
                timeout=timeout,
            )
            
            # 处理生成器
            if hasattr(response, '__iter__') and not isinstance(response, (str, dict)):
                chunks = list(response)
                response = chunks[-1] if chunks and isinstance(chunks[-1], dict) else ''.join(str(c) for c in chunks if str(c).strip() and str(c) != '{}')
            
            # 如果response是字符串，尝试用repair_json修补为JSON
            if isinstance(response, str):
                try:
                    from json_repair import repair_json
                    response = json.loads(repair_json(response))
                except ImportError:
                    pass  # 没有安装json_repair，保持字符串
                except Exception:
                    pass  # 修补失败，保持原字符串
            
            resp_time = time.time() - start_time
            text = self._extract_text_from_response(response)
            
            if not text:
                return False, "响应为空", resp_time
            if text.startswith("[错误]"):
                return False, text, resp_time
            
            # 截断
            if len(text) > 200:
                text = text[:200] + "..."
            
            return True, text, resp_time
        
        except Exception as e:
            error_msg = str(e)
            auth_type = self._identify_auth_from_error(error_msg)
            if auth_type != AuthType.NONE:
                return False, f"[{auth_type.value}] {error_msg}", None
            return False, error_msg, None
    
    def batch_test_providers(
        self,
        provider_names: Optional[List[str]] = None,
        test_prompt: str = "1+1=?",
        timeout: int = 15
    ) -> Dict[str, Tuple[bool, Optional[str], Optional[float]]]:
        """批量测试provider"""
        if provider_names is None:
            provider_names = [p.name for p in self.get_recommended_providers(10)]
        
        results = {}
        for i, name in enumerate(provider_names):
            results[name] = self.test_provider(name, test_prompt=test_prompt, timeout=timeout)
            if i < len(provider_names) - 1:
                time.sleep(0.5)
        
        return results
    
    def test_all_combinations(
        self,
        test_prompt: str = "你好",
        timeout: int = 15,
        max_workers: int = 8
    ) -> List[TestResult]:
        """并发测试所有组合"""
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        providers = self.get_working_providers()
        tasks = [(p.name, m) for p in providers for m in p.models]
        results = []
        
        def test_one(prov: str, mod: str) -> Optional[TestResult]:
            success, resp, t = self.test_provider(prov, mod, test_prompt, timeout, False)
            if success or resp:
                auth = AuthType.NONE
                if not success and resp:
                    auth = self._identify_auth_from_error(resp)
                return TestResult(prov, mod, success, resp, t, auth)
            return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(test_one, p, m): (p, m) for p, m in tasks}
            
            if use_tqdm:
                with tqdm(total=len(futures), desc="测试组合", unit="项") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        if r := future.result():
                            results.append(r)
                        pbar.update(1)
            else:
                for future in concurrent.futures.as_completed(futures):
                    if r := future.result():
                        results.append(r)
        
        return results
    
    def export_providers(self, filepath: str):
        """导出provider信息"""
        providers = self.get_all_providers()
        data = {
            'scan_time': self._last_scan_time.isoformat() if self._last_scan_time else None,
            'total': len(providers),
            'working': sum(1 for p in providers if p.working),
            'providers': [p.to_dict() for p in providers]
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def export_models(self, filepath: str):
        """导出model信息"""
        models = self.get_all_models()
        data = {
            'scan_time': self._last_scan_time.isoformat() if self._last_scan_time else None,
            'total': len(models),
            'models': [m.to_dict() for m in models]
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def export_test_results(self, results: List[TestResult], filepath: str):
        """导出Test result"""
        data = {
            'test_time': datetime.now().isoformat(),
            'total': len(results),
            'successful': sum(1 for r in results if r.success),
            'results': [r.to_dict() for r in results]
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_categorized_results(self, result_data: Dict, output_dir: Path):
        """将探测结果按类别保存到不同文件"""
        # 成功的provider-model组合
        successful = {}
        # 失败的组合（按错误类型分类）
        failed_by_auth = {}  # 需要认证
        failed_by_model_not_found = {}  # 模型不存在
        failed_by_network = {}  # 网络/连接问题
        failed_by_empty_response = {}  # 响应为空
        failed_by_other = {}  # 其他错误
        
        for prov_name, prov_info in result_data['providers'].items():
            for model_name, model_info in prov_info['models'].items():
                if model_info['success']:
                    if prov_name not in successful:
                        successful[prov_name] = {'models': {}}
                    successful[prov_name]['models'][model_name] = model_info
                else:
                    error = model_info.get('error', '')
                    
                    # 分类错误
                    if '[api_key]' in error or '[cookie]' in error or '[token]' in error:
                        target = failed_by_auth
                    elif 'Model not found' in error:
                        target = failed_by_model_not_found
                    elif '响应为空' in error:
                        target = failed_by_empty_response
                    elif 'timeout' in error.lower() or 'connection' in error.lower() or 'decode JSON' in error:
                        target = failed_by_network
                    else:
                        target = failed_by_other
                    
                    if prov_name not in target:
                        target[prov_name] = {'models': {}}
                    target[prov_name]['models'][model_name] = model_info
        
        # 保存各个分类
        categories = [
            ('successful', successful, '成功'),
            ('failed_auth_required', failed_by_auth, '需要认证'),
            ('failed_model_not_found', failed_by_model_not_found, '模型不存在'),
            ('failed_empty_response', failed_by_empty_response, '响应为空'),
            ('failed_network', failed_by_network, '网络/连接问题'),
            ('failed_other', failed_by_other, '其他错误'),
        ]
        
        for filename, data, desc in categories:
            if data:  # 只保存非空的分类
                filepath = output_dir / f"{filename}.json"
                summary = {
                    'category': desc,
                    'total_providers': len(data),
                    'total_combinations': sum(len(p['models']) for p in data.values()),
                    'providers': data
                }
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                print(f"  📁 {desc}: {filepath} ({summary['total_combinations']}个组合)")
        
        # 生成模型->providers映射文件
        if successful:
            from collections import defaultdict
            model_to_providers = defaultdict(list)
            
            for provider_name, prov_data in successful.items():
                for model_name in prov_data['models'].keys():
                    model_to_providers[model_name].append(provider_name)
            
            # 排序provider列表
            model_to_providers = {k: sorted(v) for k, v in model_to_providers.items()}
            
            mapping_file = output_dir / "models_to_providers.json"
            mapping_data = {
                'generated_at': datetime.now().isoformat(),
                'total_models': len(model_to_providers),
                'models': model_to_providers
            }
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mapping_data, f, indent=2, ensure_ascii=False)
            print(f"  🗺️  模型映射: {mapping_file} ({len(model_to_providers)}个模型)")
    
    def probe_all_working_combinations(
        self,
        test_prompt: str = "Hello",
        timeout: int = 15,
        max_workers: int = 8,
        output_file: Optional[str] = None
    ) -> Dict[str, Dict[str, any]]:
        """探测所有可用的provider和model组合
        
        Args:
            test_prompt: 测试提示词
            timeout: 每个测试的超时时间(秒)
            max_workers: 并发线程数
            output_file: 输出文件路径，如果提供则自动保存
            
        Returns:
            格式: {
                'provider_name': {
                    'working': True/False,
                    'models': {
                        'model_name': {
                            'success': True/False,
                            'response_time': float,
                            'error': str (if failed)
                        }
                    }
                }
            }
        """
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        providers = self.get_all_providers()
        # 构建所有provider-model组合
        tasks = []
        for p in providers:
            if p.working:
                # BlackboxPro特殊处理：只能使用空字符串model
                if p.name == 'BlackboxPro':
                    tasks.append((p.name, ''))
                elif p.models:
                    # 有明确模型列表的provider，过滤空字符串
                    valid_models = [m for m in p.models if m and isinstance(m, str)]
                    for model in valid_models:
                        tasks.append((p.name, model))
                else:
                    # 没有模型列表的provider，测试默认模型
                    tasks.append((p.name, None))
        
        results_map = {}
        
        def test_one(provider_name: str, model_name: Optional[str]) -> Tuple[str, Optional[str], bool, Optional[float], Optional[str]]:
            """测试单个组合，返回(provider, model, success, time, error)"""
            success, resp, resp_time = self.test_provider(
                provider_name, 
                model_name, 
                test_prompt, 
                timeout, 
                verbose=False
            )
            error = None if success else resp
            return provider_name, model_name, success, resp_time, error
        
        # 并发测试
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(test_one, prov, mod): (prov, mod) for prov, mod in tasks}
            
            iterator = concurrent.futures.as_completed(futures)
            if use_tqdm:
                iterator = tqdm(iterator, total=len(futures), desc="探测provider-model组合", unit="项")
            
            for future in iterator:
                provider_name, model_name, success, resp_time, error = future.result()
                
                # 初始化provider条目
                if provider_name not in results_map:
                    results_map[provider_name] = {
                        'working': False,
                        'models': {}
                    }
                
                # 记录模型Test result
                model_key = model_name if model_name else '__default__'
                results_map[provider_name]['models'][model_key] = {
                    'success': success,
                    'response_time': resp_time
                }
                if error:
                    results_map[provider_name]['models'][model_key]['error'] = error
                
                # 如果有任何一个模型成功，标记provider为working
                if success:
                    results_map[provider_name]['working'] = True
        
        # 构建最终结果
        result_data = {
            'probe_time': datetime.now().isoformat(),
            'total_providers': len(results_map),
            'working_providers': sum(1 for p in results_map.values() if p['working']),
            'total_combinations': len(tasks),
            'successful_combinations': sum(
                1 for p in results_map.values() 
                for m in p['models'].values() 
                if m['success']
            ),
            'providers': results_map
        }
        
        # 保存完整结果到文件
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存完整结果
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            print(f"✅ 完整结果: {output_path}")
            
            # 按成功/失败/错误类型分类保存
            self._save_categorized_results(result_data, output_path.parent)
        
        return result_data
    
    def print_summary(self, include_real_test: bool = False):
        """打印摘要"""
        providers = self.get_all_providers()
        working = [p for p in providers if p.working]
        recommended = self.get_recommended_providers(5)
        
        print(f"\n{'='*70}")
        print("G4FAdmin 状态摘要")
        print(f"{'='*70}")
        print(f"📊 总Provider数: {len(providers)}")
        print(f"✅ 可用Provider: {len(working)}")
        print(f"❌ 不可用Provider: {len(providers) - len(working)}")
        
        if recommended:
            print(f"\n🎯 推荐Provider (前5):")
            for i, p in enumerate(recommended, 1):
                features = []
                if p.supports_stream:
                    features.append("流式")
                if p.supports_message_history:
                    features.append("历史")
                if p.models:
                    features.append(f"{len(p.models)}模型")
                feat_str = ", ".join(features) or "基础"
                print(f"  {i}. {p.name:25s} [{feat_str}]")
        
        if include_real_test and recommended:
            print(f"\n🧪 真实测试 (前3个推荐):")
            for i, p in enumerate(recommended[:3], 1):
                success, resp, t = self.test_provider(p.name, test_prompt="1+1=?", timeout=15)
                status = "✅" if success else "❌"
                time_str = f"{t:.2f}s" if t else "N/A"
                result = (resp[:50] if resp else "")
                print(f"  {i}. {status} {p.name:25s} [{time_str}] {result}")
        
        models = self.get_all_models()
        if models:
            print(f"\n📦 总Model数: {len(models)}")
            print("\n🔥 热门Model (前10):")
            sorted_models = sorted(models, key=lambda m: len(m.providers), reverse=True)
            for i, m in enumerate(sorted_models[:10], 1):
                print(f"  {i}. {m.name:30s} ({len(m.providers)} providers)")
        
        print(f"\n{'='*70}\n")
    
    def _chat_internal(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str],
        model: Optional[str],
        auto_select: bool,
        timeout: int
    ) -> Tuple[bool, str, Optional[str], Optional[str]]:
        """内部聊天实现（非流式）"""
        # 获取候选组合
        candidates = self._get_chat_candidates(provider, model)
        
        if not candidates:
            # 给出更明确的错误信息
            if model is not None:
                error_msg = f"找不到支持模型 '{model}' 的可用provider。请使用 'g4fadmin --find {model}' 查找支持该模型的providers，或使用 'g4fadmin --list-models' 查看所有可用模型。"
            else:
                error_msg = "没有可用的provider"
            return False, error_msg, None, None
        
        # 尝试候选组合
        last_error = ""
        for i, candidate in enumerate(candidates):
            try:
                prov_name = candidate['provider']
                model_name = candidate['model']
                
                if not hasattr(self.g4f.Provider, prov_name):
                    continue
                
                provider_class = getattr(self.g4f.Provider, prov_name)
                
                # 检查是否需要认证
                if hasattr(provider_class, 'needs_auth') and provider_class.needs_auth:
                    last_error = f"{prov_name} 需要认证。请使用 'g4fadmin --cookie-providers' 查看配置方法，或选择其他模型。"
                    logger.debug(f"Provider {prov_name} 需要认证，跳过")
                    if not auto_select or i >= len(candidates) - 1:
                        return False, last_error, prov_name, model_name
                    continue
                
                # BlackboxPro特殊处理
                if prov_name == 'BlackboxPro':
                    model_name = ''
                
                # 如果没有指定model，自动选择
                if model_name is None:
                    model_name = self._auto_select_model(provider_class)
                
                # 调用g4f（非流式）
                response = self.g4f.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                    provider=provider_class,
                    timeout=timeout,
                    stream=False
                )
                
                # 处理响应
                if hasattr(response, '__iter__') and not isinstance(response, (str, dict)):
                    chunks = list(response)
                    response = chunks[-1] if chunks and isinstance(chunks[-1], dict) else ''.join(str(c) for c in chunks if str(c).strip() and str(c) != '{}')
                
                # 如果是字符串，尝试修复JSON
                if isinstance(response, str):
                    try:
                        from json_repair import repair_json
                        response = json.loads(repair_json(response))
                    except:
                        pass
                
                result = self._extract_text_from_response(response)
                
                if not result:
                    last_error = "响应为空"
                    if not auto_select or i >= len(candidates) - 1:
                        return False, last_error, prov_name, model_name
                    continue
                
                return True, result, prov_name, model_name
                
            except Exception as e:
                last_error = str(e)
                logger.debug(f"Provider {candidate['provider']} 失败: {last_error}")
                
                # 检查是否是认证错误
                if 'auth' in last_error.lower() or 'cookie' in last_error.lower() or 'MissingAuthError' in last_error:
                    last_error = f"{candidate['provider']} 需要认证。请使用 'g4fadmin --cookie-providers' 查看配置方法，或选择其他模型。"
                
                if not auto_select or i >= len(candidates) - 1:
                    return False, last_error, candidate['provider'], candidate.get('model')
                
                continue
        
        return False, last_error, None, None
    
    def _chat_stream(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str],
        model: Optional[str],
        auto_select: bool,
        timeout: int
    ):
        """内部聊天实现（流式）- 生成器"""
        # 获取候选组合
        candidates = self._get_chat_candidates(provider, model)
        
        if not candidates:
            # 给出更明确的错误信息
            if model is not None:
                error_msg = f"找不到支持模型 '{model}' 的可用provider。请使用 'g4fadmin --find {model}' 查找支持该模型的providers，或使用 'g4fadmin --list-models' 查看所有可用模型。"
            else:
                error_msg = "没有可用的provider"
            yield ("error", error_msg, None, None)
            return
        
        # 尝试候选组合
        last_error = ""
        for i, candidate in enumerate(candidates):
            try:
                prov_name = candidate['provider']
                model_name = candidate['model']
                
                if not hasattr(self.g4f.Provider, prov_name):
                    continue
                
                provider_class = getattr(self.g4f.Provider, prov_name)
                
                # 检查是否需要认证
                if hasattr(provider_class, 'needs_auth') and provider_class.needs_auth:
                    last_error = f"{prov_name} 需要认证。请使用 'g4fadmin --cookie-providers' 查看配置方法，或选择其他模型。"
                    logger.debug(f"Provider {prov_name} 需要认证，跳过")
                    if not auto_select or i >= len(candidates) - 1:
                        yield ("error", last_error, prov_name, model_name)
                        return
                    continue
                
                # BlackboxPro特殊处理
                if prov_name == 'BlackboxPro':
                    model_name = ''
                
                # 如果没有指定model，自动选择
                if model_name is None:
                    model_name = self._auto_select_model(provider_class)
                
                # 调用g4f（流式）
                response = self.g4f.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                    provider=provider_class,
                    timeout=timeout,
                    stream=True
                )
                
                # yield chunks - 提取文本内容
                for chunk in response:
                    if not chunk:
                        continue
                    
                    # 流式chunk是特殊格式: {'choices': [{'delta': {'content': '...'}}]}
                    text = None
                    if isinstance(chunk, dict):
                        # 提取 delta.content
                        if 'choices' in chunk and chunk['choices']:
                            delta = chunk['choices'][0].get('delta', {})
                            text = delta.get('content', '')
                    elif isinstance(chunk, str):
                        text = chunk
                    # 其他类型直接跳过，不要转str
                    
                    # 只yield非空文本内容
                    if text:
                        yield text
                
                # 成功完成
                yield ("success", prov_name, model_name)
                return
                
            except Exception as e:
                last_error = str(e)
                logger.debug(f"Provider {candidate['provider']} 失败: {last_error}")
                
                # 检查是否是认证错误
                if 'auth' in last_error.lower() or 'cookie' in last_error.lower() or 'MissingAuthError' in last_error:
                    last_error = f"{candidate['provider']} 需要认证。请使用 'g4fadmin --cookie-providers' 查看配置方法，或选择其他模型。"
                
                if not auto_select or i >= len(candidates) - 1:
                    yield ("error", last_error, candidate['provider'], candidate.get('model'))
                    return
                
                continue
        
        # 所有候选都失败
        yield ("error", last_error, None, None)
    
    def _get_chat_candidates(self, provider: Optional[str], model: Optional[str]) -> List[Dict]:
        """获取聊天候选组合"""
        # 如果指定了provider
        if provider is not None:
            # 直接使用指定的provider和model
            return [{'provider': provider, 'model': model, 'time': 0}]
        
        # 如果没有指定provider但指定了model，在successful.json中查找支持该model的provider
        if model is not None:
            output_dir = Path("output")
            successful_file = output_dir / "successful.json"
            
            candidates = []
            if successful_file.exists():
                try:
                    with open(successful_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 查找支持该model的providers，按响应时间排序
                    for prov_name, prov_data in data.get('providers', {}).items():
                        if model in prov_data.get('models', {}):
                            model_info = prov_data['models'][model]
                            if model_info.get('success'):
                                candidates.append({
                                    'provider': prov_name,
                                    'model': model,
                                    'time': model_info.get('response_time', 999)
                                })
                    
                    if candidates:
                        candidates.sort(key=lambda x: x['time'])
                        return candidates
                except Exception as e:
                    logger.warning(f"无法读取successful.json: {e}")
            
            # 如果在successful.json中找不到，尝试从所有provider中查找
            # 这包括可能需要认证但用户可能已配置的provider
            logger.info(f"在successful.json中未找到模型 {model}，尝试从所有providers查找...")
            
            # 查找所有声称支持该模型的provider
            potential_providers = []
            for provider_name in dir(self.g4f.Provider):
                if provider_name.startswith('_'):
                    continue
                
                try:
                    provider_class = getattr(self.g4f.Provider, provider_name)
                    if not hasattr(provider_class, '__mro__'):
                        continue
                    
                    # 检查是否有models属性
                    if hasattr(provider_class, 'models'):
                        models = provider_class.models
                        if isinstance(models, list) and model in models:
                            potential_providers.append({
                                'provider': provider_name,
                                'model': model,
                                'time': 999
                            })
                except:
                    continue
            
            if potential_providers:
                logger.info(f"找到 {len(potential_providers)} 个声称支持 {model} 的providers，将尝试使用")
                return potential_providers
            
            # 完全找不到支持该模型的provider，返回空列表
            # 这会导致错误提示而不是回退到其他模型
            logger.warning(f"找不到任何支持模型 {model} 的provider")
            return []
        
        # 既没有指定provider也没有指定model，自动选择
        # 优先使用successful.json中的组合
        output_dir = Path("output")
        successful_file = output_dir / "successful.json"
        
        candidates = []
        if successful_file.exists():
            try:
                with open(successful_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取所有成功的provider-model组合，按响应时间排序
                for prov_name, prov_data in data.get('providers', {}).items():
                    for model_name, model_info in prov_data.get('models', {}).items():
                        if model_info.get('success'):
                            candidates.append({
                                'provider': prov_name,
                                'model': model_name,
                                'time': model_info.get('response_time', 999)
                            })
                
                candidates.sort(key=lambda x: x['time'])
            except Exception as e:
                logger.warning(f"无法读取successful.json: {e}")
        
        # 如果没有successful.json，使用推荐provider
        if not candidates:
            recommended = self.get_recommended_providers(5)
            for p in recommended:
                if p.models:
                    candidates.append({
                        'provider': p.name,
                        'model': p.models[0],
                        'time': 999
                    })
                else:
                    candidates.append({
                        'provider': p.name,
                        'model': None,
                        'time': 999
                    })
        
        return candidates
    
    def _auto_select_model(self, provider_class) -> str:
        """自动选择模型"""
        if hasattr(provider_class, 'default_model'):
            return provider_class.default_model
        elif hasattr(provider_class, 'models') and provider_class.models:
            models_list = provider_class.models
            if isinstance(models_list, list):
                valid = [m for m in models_list if m]
                return valid[0] if valid else 'auto'
            else:
                return 'auto'
        else:
            return 'auto'
    
    def chat(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = False,
        history: Optional[List[Dict[str, str]]] = None,
        auto_select: bool = True,
        timeout: int = 30
    ):
        """聊天接口，封装g4f的ChatCompletion
        
        Args:
            message: 用户消息
            provider: 指定provider名称，None则自动选择
            model: 指定模型名称，None则使用provider默认
            stream: 是否使用流式输出
            history: 消息历史 [{"role": "user", "content": "..."}, ...]
            auto_select: 如果指定的组合失败，是否自动尝试其他组合
            timeout: 超时时间(秒)
            
        Returns:
            stream=False: (success, response, used_provider, used_model)
            stream=True: generator yielding chunks (最后yield元组信息)
        """
        # 构建消息列表
        messages = history.copy() if history else []
        messages.append({"role": "user", "content": message})
        
        if stream:
            return self._chat_stream(messages, provider, model, auto_select, timeout)
        else:
            return self._chat_internal(messages, provider, model, auto_select, timeout)

