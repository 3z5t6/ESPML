#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
from typing import Dict, Any, Optional

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class YamlParser:
    
    @staticmethod
    def parse(path: str, mode: Optional[str] = None) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise ValueError(f'Configuration file does not exist: {path}')
            
        with open(path, 'rt', encoding='utf-8') as fh:
            config = yaml.load(fh, Loader=Loader)
            if mode is not None:
                config = config[mode]
            fh.seek(0)
            file_lines = fh.readlines()
            
            for line in file_lines:
                try:
                    parts = line.strip().split('\n')[0].split(' ')
                    if parts[0] == 'AuthorName:':
                        if 'AuthorName' in config and str(config['AuthorName']) != parts[1] and parts[1].isdigit():
                            config['AuthorName'] = parts[1].replace("'", '')
                        break
                except Exception:
                    continue
            
            return config
    
    @staticmethod
    def dump(path: str, config: Dict[str, Any]) -> None:
        with open(path, 'w', encoding='utf-8') as fh:
            yaml.safe_dump(config, fh, allow_unicode=True)