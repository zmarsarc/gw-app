import logging
import os
from logging import handlers

# 日志级别关系映射
level_relations = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'

log_dir_path = os.path.dirname(os.path.dirname(__file__))

default_filename = os.path.join(log_dir_path, 'logs/app.log')

file_rotated_filename = os.path.join(log_dir_path, 'logs/app_file_rotated.log')
time_rotated_filename = os.path.join(log_dir_path, 'logs/app_time_rotated.log')


class CustomLogger():
    def __init__(
            self,
            filename=None,
            level='debug',
            fmt=None,
            stream_handler=False,
            **kwargs
    ):
        self.filename = filename
        self.level = level
        self.fmt = fmt,
        self.stream_handler = stream_handler
        self.logger = logging.getLogger(self.filename)
        self.format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(level_relations.get(self.level))  # 设置日志级别
        self.logger.propagate = False

        f_dir, f_name = os.path.split(self.filename)
        if f_dir:
            os.makedirs(f_dir, exist_ok=True)

    def add_handler(self, handler_type='file_handler', **kwargs):

        handler = None

        if handler_type == 'file_handler':
            handler = logging.FileHandler(filename=self.filename, encoding='utf-8')
        elif handler_type == 'file_rotated':
            handler = handlers.RotatingFileHandler(
                filename=self.filename,
                maxBytes=kwargs.get('maxBytes', 1024 * 1024 * 100),  # 10MB大小转存
                backupCount=kwargs.get('backupCount', 20),
                encoding='utf-8'
            )
        elif handler_type == 'time_rotated':
            handler = handlers.TimedRotatingFileHandler(
                filename=self.filename,
                when=kwargs.get('when', 'D'),  # 默认按天转存
                backupCount=kwargs.get('backupCount', 60),
                encoding='utf-8'
            )

        handler.setFormatter(self.format_str)

        if not self.logger.handlers:
            self.logger.addHandler(handler)

        if self.stream_handler:
            sh = logging.StreamHandler()
            sh.setFormatter(self.format_str)
            if self.logger.handlers and len(self.logger.handlers) == 1:
                self.logger.addHandler(sh)  # 把对象加到logger里
        return self.logger


def get_logger(
        filename=default_filename,
        level='debug',
        fmt=fmt,
        stream_handler=True,
):
    logger = CustomLogger(
        filename=filename,
        level=level,
        fmt=fmt,
        stream_handler=stream_handler
    )
    return logger.add_handler()


def get_file_rotated_logger(
        filename=file_rotated_filename,
        level='debug',
        fmt=fmt,
        stream_handler=False,
        maxBytes=1024 * 1024 * 100,
        backupCount=20
):
    logger = CustomLogger(
        filename=filename,
        level=level,
        fmt=fmt,
        stream_handler=stream_handler
    )
    return logger.add_handler(handler_type='file_rotated', maxBytes=maxBytes, backupCount=backupCount)


def get_time_rotated_logger(
        filename=time_rotated_filename,
        level='debug',
        fmt=fmt,
        stream_handler=False,
        when='D',
        backupCount=20
):
    logger = CustomLogger(
        filename=filename,
        level=level,
        fmt=fmt,
        stream_handler=stream_handler
    )
    return logger.add_handler(handler_type='file_rotated', when=when, backupCount=backupCount)


if __name__ == '__main__':
    logger1 = get_logger(stream_handler=True)

    logger1.info('info\n新起一行')
    # logger.warning('警告')
    # logger.error('报错')
    # logger.critical('严重')
