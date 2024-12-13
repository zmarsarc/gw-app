import importlib
import os


def module_location(target=None):
    '''
    get loc dir and module
    :param target: py module name eg: model_handel, check_data
    :return: '.service/xxx-xxx' loc dir or '.'
    '''
    try:
        prepath = os.getenv('prepath')
        if not prepath:
            raise ValueError('prepath not found')
        def get_env_service():
            for service_name in os.listdir('./services'):
                if service_name.startswith(prepath):
                    return service_name

        if not target:
            return f'./services/{get_env_service()}', None
        if service_name := get_env_service():
            module_path = f'services.{service_name}.{target}'
            target_module = importlib.import_module(module_path)
            models_base_path = f'./services/{service_name}'
        else:
            target_module = importlib.import_module(target)
            models_base_path = '.'
        return models_base_path, target_module
    except Exception as e:
        print(e)
        return None
