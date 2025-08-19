import sys, traceback
from src.logger import logger
def error_message_detail(error_message, error_detail:sys):
    _,_,exc_tab = error_detail.exc_info()
    file_name= exc_tab.tb_frame.f_code.co_filename
    line_no= exc_tab.tb_lineno
    return f"Error occurred in script: {file_name}, line: {line_no}, message: {error_message}"


class CustomException(Exception):
    def __init__(self,error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message= error_message_detail(error_message, error_detail)

        logger.error(self.error_message)

    def __str__(self):
        return self.error_message

