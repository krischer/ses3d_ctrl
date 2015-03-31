from __future__ import absolute_import

from .site_config import SiteConfig
from .local_gcc import LocalGCC
from .supermuc import SuperMuc

available_sites = {_i.site_name: _i for _i in SiteConfig.__subclasses__()}
