class AppException(Exception):
    """
    Application-wide exception with optional detail.
    Attributes:
      - message: user-facing message
      - error_detail: original exception or traceback
    """
    def __init__(self, message: str, error_detail: Exception = None):
        super().__init__(message)
        self.message = message
        self.error_detail = error_detail

    def __str__(self):
        return f"AppException: {self.message}"