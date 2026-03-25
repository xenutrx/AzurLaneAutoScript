# Lightweight test for Siren Detector ignore behavior
from types import SimpleNamespace
import logging

logger = logging.getLogger('test')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

class Dummy:
    pass

def run_test():
    dummy = Dummy()
    cfg = SimpleNamespace()
    cfg.OpsiMeowfficerFarming_StayInZone = True
    dummy.config = cfg

    # simulate entering search
    siren_search_enabled = True
    if siren_search_enabled:
        if not getattr(dummy, '_siren_search_ignore_stay_in_zone', False):
            dummy._siren_search_ignore_stay_in_zone = True
            if dummy.config.OpsiMeowfficerFarming_StayInZone:
                logger.info('探测装置搜索模式：临时忽略指定海域计划作战（不修改配置）')

    logger.info('State during search: StayInZone=%s, ignore_flag=%s',
                dummy.config.OpsiMeowfficerFarming_StayInZone,
                getattr(dummy, '_siren_search_ignore_stay_in_zone', False))

    # simulate search complete -> remove ignore flag but do not change config
    if getattr(dummy, '_siren_search_ignore_stay_in_zone', False):
        try:
            delattr(dummy, '_siren_search_ignore_stay_in_zone')
        except Exception:
            pass
        logger.info('探测装置搜索完成：恢复指定海域计划作战（配置未被关闭）')

    logger.info('Final state: StayInZone=%s, ignore_flag_present=%s',
                dummy.config.OpsiMeowfficerFarming_StayInZone,
                hasattr(dummy, '_siren_search_ignore_stay_in_zone'))

if __name__ == '__main__':
    run_test()
