# tests/config/test_yaml_parser.py
import pytest
from pathlib import Path
import yaml
from espml.config.yaml_parser import load_yaml_config, ConfigError # 导入被测函数

# 使用 pytest fixture 创建临时 YAML 文件
@pytest.fixture
def valid_yaml_file(tmp_path: Path) -> Path:
    content = {
        'SectionA': {'param1': 'value1', 'param2': 123},
        'SectionB': ['item1', 'item2'],
        'TopLevelParam': True
    }
    file_path = tmp_path / "valid.yaml"
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(content, f)
    return file_path

@pytest.fixture
def invalid_yaml_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "invalid.yaml"
    file_path.write_text("SectionA: param1: value1\n  param2: 123", encoding='utf-8') # 格式错误
    return file_path

@pytest.fixture
def non_dict_yaml_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "nondict.yaml"
    file_path.write_text("- item1\n- item2", encoding='utf-8') # 顶层是列表
    return file_path

def test_load_valid_yaml_full(valid_yaml_file: Path):
    """测试加载完整的有效 YAML 文件"""
    config = load_yaml_config(valid_yaml_file)
    assert isinstance(config, dict)
    assert 'SectionA' in config
    assert config['SectionA']['param1'] == 'value1'
    assert 'TopLevelParam' in config
    assert config['TopLevelParam'] is True

def test_load_valid_yaml_section(valid_yaml_file: Path):
    """测试加载有效 YAML 文件的特定部分"""
    section_a = load_yaml_config(valid_yaml_file, section='SectionA')
    assert isinstance(section_a, dict)
    assert section_a['param1'] == 'value1'
    assert section_a['param2'] == 123

def test_load_non_existent_file(tmp_path: Path):
    """测试加载不存在的文件"""
    non_existent_path = tmp_path / "not_found.yaml"
    with pytest.raises(FileNotFoundError):
        load_yaml_config(non_existent_path)

def test_load_invalid_yaml(invalid_yaml_file: Path):
    """测试加载格式无效的 YAML 文件"""
    with pytest.raises(ConfigError, match="YAML 格式错误"):
        load_yaml_config(invalid_yaml_file)

def test_load_non_dict_yaml(non_dict_yaml_file: Path):
    """测试加载顶层不是字典的 YAML 文件"""
    with pytest.raises(ConfigError, match="顶层结构必须是字典"):
        load_yaml_config(non_dict_yaml_file)

def test_load_non_existent_section(valid_yaml_file: Path):
    """测试加载存在的 YAML 文件中不存在的部分"""
    with pytest.raises(ConfigError, match="未找到顶级部分: 'SectionC'"):
        load_yaml_config(valid_yaml_file, section='SectionC')

def test_load_non_dict_section(valid_yaml_file: Path):
    """测试加载一个值不是字典的 section (应直接返回值)"""
    section_b = load_yaml_config(valid_yaml_file, section='SectionB')
    assert section_b == ['item1', 'item2']
    top_level = load_yaml_config(valid_yaml_file, section='TopLevelParam')
    assert top_level is True