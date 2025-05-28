class AppException(Exception):
    def __init__(self,
                 message: str,
                 status_code: int = 400,
                 error_detail: Exception = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_detail = error_detail

    def __str__(self):
        return f"{self.status_code} - {self.message}"
