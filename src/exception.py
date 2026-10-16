
import sys


def error_message_detail(error: Exception) -> str:
    _, _, exc_tb = sys.exc_info()

    if exc_tb is None:
        return str(error)

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error in [{file_name}] "
        f"at line [{line_number}]: {error}"
    )


class CustomerChurnException(Exception):
    def __init__(self, error: Exception):
        super().__init__(error_message_detail(error))
