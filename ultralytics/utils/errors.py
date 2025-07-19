<<<<<<< HEAD
# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import emojis


class HUBModelError(Exception):
    """
    Custom exception class for handling errors related to model fetching in Ultralytics YOLO.

    This exception is raised when a requested model is not found or cannot be retrieved.
    The message is also processed to include emojis for better user experience.

    Attributes:
        message (str): The error message displayed when the exception is raised.

    Note:
        The message is automatically processed through the 'emojis' function from the 'ultralytics.utils' package.
    """

    def __init__(self, message='Model not found. Please check model URL and try again.'):
        """Create an exception for when a model is not found."""
        super().__init__(emojis(message))
=======
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import emojis


class HUBModelError(Exception):
    """
    Exception raised when a model cannot be found or retrieved from Ultralytics HUB.

    This custom exception is used specifically for handling errors related to model fetching in Ultralytics YOLO.
    The error message is processed to include emojis for better user experience.

    Attributes:
        message (str): The error message displayed when the exception is raised.

    Methods:
        __init__: Initialize the HUBModelError with a custom message.

    Examples:
        >>> try:
        ...     # Code that might fail to find a model
        ...     raise HUBModelError("Custom model not found message")
        ... except HUBModelError as e:
        ...     print(e)  # Displays the emoji-enhanced error message
    """

    def __init__(self, message: str = "Model not found. Please check model URL and try again."):
        """
        Initialize a HUBModelError exception.

        This exception is raised when a requested model is not found or cannot be retrieved from Ultralytics HUB.
        The message is processed to include emojis for better user experience.

        Args:
            message (str, optional): The error message to display when the exception is raised.

        Examples:
            >>> try:
            ...     raise HUBModelError("Custom model error message")
            ... except HUBModelError as e:
            ...     print(e)
        """
        super().__init__(emojis(message))
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
