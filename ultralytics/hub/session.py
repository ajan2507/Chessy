<<<<<<< HEAD
# Ultralytics YOLO 🚀, AGPL-3.0 license

import signal
import sys
from pathlib import Path
from time import sleep

import requests

from ultralytics.hub.utils import HUB_API_ROOT, HUB_WEB_ROOT, PREFIX, smart_request
from ultralytics.utils import LOGGER, __version__, checks, emojis, is_colab, threaded
from ultralytics.utils.errors import HUBModelError

AGENT_NAME = f'python-{__version__}-colab' if is_colab() else f'python-{__version__}-local'


class HUBTrainingSession:
    """
    HUB training session for Ultralytics HUB YOLO models. Handles model initialization, heartbeats, and checkpointing.

    Args:
        url (str): Model identifier used to initialize the HUB training session.

    Attributes:
        agent_id (str): Identifier for the instance communicating with the server.
        model_id (str): Identifier for the YOLO model being trained.
        model_url (str): URL for the model in Ultralytics HUB.
        api_url (str): API URL for the model in Ultralytics HUB.
        auth_header (dict): Authentication header for the Ultralytics HUB API requests.
        rate_limits (dict): Rate limits for different API calls (in seconds).
        timers (dict): Timers for rate limiting.
        metrics_queue (dict): Queue for the model's metrics.
        model (dict): Model data fetched from Ultralytics HUB.
        alive (bool): Indicates if the heartbeat loop is active.
    """

    def __init__(self, url):
        """
        Initialize the HUBTrainingSession with the provided model identifier.

        Args:
            url (str): Model identifier used to initialize the HUB training session.
                         It can be a URL string or a model key with specific format.

        Raises:
            ValueError: If the provided model identifier is invalid.
            ConnectionError: If connecting with global API key is not supported.
        """

        from ultralytics.hub.auth import Auth

        # Parse input
        if url.startswith(f'{HUB_WEB_ROOT}/models/'):
            url = url.split(f'{HUB_WEB_ROOT}/models/')[-1]
        if [len(x) for x in url.split('_')] == [42, 20]:
            key, model_id = url.split('_')
        elif len(url) == 20:
            key, model_id = '', url
        else:
            raise HUBModelError(f"model='{url}' not found. Check format is correct, i.e. "
                                f"model='{HUB_WEB_ROOT}/models/MODEL_ID' and try again.")

        # Authorize
        auth = Auth(key)
        self.agent_id = None  # identifies which instance is communicating with server
        self.model_id = model_id
        self.model_url = f'{HUB_WEB_ROOT}/models/{model_id}'
        self.api_url = f'{HUB_API_ROOT}/v1/models/{model_id}'
        self.auth_header = auth.get_auth_header()
        self.rate_limits = {'metrics': 3.0, 'ckpt': 900.0, 'heartbeat': 300.0}  # rate limits (seconds)
        self.timers = {}  # rate limit timers (seconds)
        self.metrics_queue = {}  # metrics queue
        self.model = self._get_model()
        self.alive = True
        self._start_heartbeat()  # start heartbeats
        self._register_signal_handlers()
        LOGGER.info(f'{PREFIX}View model at {self.model_url} 🚀')

    def _register_signal_handlers(self):
        """Register signal handlers for SIGTERM and SIGINT signals to gracefully handle termination."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """
        Handle kill signals and prevent heartbeats from being sent on Colab after termination.

        This method does not use frame, it is included as it is passed by signal.
        """
        if self.alive is True:
            LOGGER.info(f'{PREFIX}Kill signal received! ❌')
            self._stop_heartbeat()
            sys.exit(signum)

    def _stop_heartbeat(self):
        """Terminate the heartbeat loop."""
        self.alive = False

    def upload_metrics(self):
        """Upload model metrics to Ultralytics HUB."""
        payload = {'metrics': self.metrics_queue.copy(), 'type': 'metrics'}
        smart_request('post', self.api_url, json=payload, headers=self.auth_header, code=2)

    def _get_model(self):
        """Fetch and return model data from Ultralytics HUB."""
        api_url = f'{HUB_API_ROOT}/v1/models/{self.model_id}'

        try:
            response = smart_request('get', api_url, headers=self.auth_header, thread=False, code=0)
            data = response.json().get('data', None)

            if data.get('status', None) == 'trained':
                raise ValueError(emojis(f'Model is already trained and uploaded to {self.model_url} 🚀'))

            if not data.get('data', None):
                raise ValueError('Dataset may still be processing. Please wait a minute and try again.')  # RF fix
            self.model_id = data['id']

            if data['status'] == 'new':  # new model to start training
                self.train_args = {
                    'batch': data['batch_size'],  # note HUB argument is slightly different
                    'epochs': data['epochs'],
                    'imgsz': data['imgsz'],
                    'patience': data['patience'],
                    'device': data['device'],
                    'cache': data['cache'],
                    'data': data['data']}
                self.model_file = data.get('cfg') or data.get('weights')  # cfg for pretrained=False
                self.model_file = checks.check_yolov5u_filename(self.model_file, verbose=False)  # YOLOv5->YOLOv5u
            elif data['status'] == 'training':  # existing model to resume training
                self.train_args = {'data': data['data'], 'resume': True}
                self.model_file = data['resume']

            return data
        except requests.exceptions.ConnectionError as e:
            raise ConnectionRefusedError('ERROR: The HUB server is not online. Please try again later.') from e
        except Exception:
            raise

    def upload_model(self, epoch, weights, is_best=False, map=0.0, final=False):
        """
        Upload a model checkpoint to Ultralytics HUB.

        Args:
            epoch (int): The current training epoch.
            weights (str): Path to the model weights file.
            is_best (bool): Indicates if the current model is the best one so far.
            map (float): Mean average precision of the model.
            final (bool): Indicates if the model is the final model after training.
        """
        if Path(weights).is_file():
            with open(weights, 'rb') as f:
                file = f.read()
        else:
            LOGGER.warning(f'{PREFIX}WARNING ⚠️ Model upload issue. Missing model {weights}.')
            file = None
        url = f'{self.api_url}/upload'
        # url = 'http://httpbin.org/post'  # for debug
        data = {'epoch': epoch}
        if final:
            data.update({'type': 'final', 'map': map})
            filesize = Path(weights).stat().st_size
            smart_request('post',
                          url,
                          data=data,
                          files={'best.pt': file},
                          headers=self.auth_header,
                          retry=10,
                          timeout=3600,
                          thread=False,
                          progress=filesize,
                          code=4)
        else:
            data.update({'type': 'epoch', 'isBest': bool(is_best)})
            smart_request('post', url, data=data, files={'last.pt': file}, headers=self.auth_header, code=3)

    @threaded
    def _start_heartbeat(self):
        """Begin a threaded heartbeat loop to report the agent's status to Ultralytics HUB."""
        while self.alive:
            r = smart_request('post',
                              f'{HUB_API_ROOT}/v1/agent/heartbeat/models/{self.model_id}',
                              json={
                                  'agent': AGENT_NAME,
                                  'agentId': self.agent_id},
                              headers=self.auth_header,
                              retry=0,
                              code=5,
                              thread=False)  # already in a thread
            self.agent_id = r.json().get('data', {}).get('agentId', None)
            sleep(self.rate_limits['heartbeat'])
=======
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import shutil
import threading
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import requests

from ultralytics import __version__
from ultralytics.hub.utils import HELP_MSG, HUB_WEB_ROOT, PREFIX
from ultralytics.utils import IS_COLAB, LOGGER, SETTINGS, TQDM, checks, emojis
from ultralytics.utils.errors import HUBModelError

AGENT_NAME = f"python-{__version__}-colab" if IS_COLAB else f"python-{__version__}-local"


class HUBTrainingSession:
    """
    HUB training session for Ultralytics HUB YOLO models.

    This class encapsulates the functionality for interacting with Ultralytics HUB during model training, including
    model creation, metrics tracking, and checkpoint uploading.

    Attributes:
        model_id (str): Identifier for the YOLO model being trained.
        model_url (str): URL for the model in Ultralytics HUB.
        rate_limits (Dict[str, int]): Rate limits for different API calls in seconds.
        timers (Dict[str, Any]): Timers for rate limiting.
        metrics_queue (Dict[str, Any]): Queue for the model's metrics.
        metrics_upload_failed_queue (Dict[str, Any]): Queue for metrics that failed to upload.
        model (Any): Model data fetched from Ultralytics HUB.
        model_file (str): Path to the model file.
        train_args (Dict[str, Any]): Arguments for training the model.
        client (Any): Client for interacting with Ultralytics HUB.
        filename (str): Filename of the model.

    Examples:
        Create a training session with a model URL
        >>> session = HUBTrainingSession("https://hub.ultralytics.com/models/example-model")
        >>> session.upload_metrics()
    """

    def __init__(self, identifier: str):
        """
        Initialize the HUBTrainingSession with the provided model identifier.

        Args:
            identifier (str): Model identifier used to initialize the HUB training session. It can be a URL string
                or a model key with specific format.

        Raises:
            ValueError: If the provided model identifier is invalid.
            ConnectionError: If connecting with global API key is not supported.
            ModuleNotFoundError: If hub-sdk package is not installed.
        """
        from hub_sdk import HUBClient

        self.rate_limits = {"metrics": 3, "ckpt": 900, "heartbeat": 300}  # rate limits (seconds)
        self.metrics_queue = {}  # holds metrics for each epoch until upload
        self.metrics_upload_failed_queue = {}  # holds metrics for each epoch if upload failed
        self.timers = {}  # holds timers in ultralytics/utils/callbacks/hub.py
        self.model = None
        self.model_url = None
        self.model_file = None
        self.train_args = None

        # Parse input
        api_key, model_id, self.filename = self._parse_identifier(identifier)

        # Get credentials
        active_key = api_key or SETTINGS.get("api_key")
        credentials = {"api_key": active_key} if active_key else None  # set credentials

        # Initialize client
        self.client = HUBClient(credentials)

        # Load models
        try:
            if model_id:
                self.load_model(model_id)  # load existing model
            else:
                self.model = self.client.model()  # load empty model
        except Exception:
            if identifier.startswith(f"{HUB_WEB_ROOT}/models/") and not self.client.authenticated:
                LOGGER.warning(
                    f"{PREFIX}Please log in using 'yolo login API_KEY'. "
                    "You can find your API Key at: https://hub.ultralytics.com/settings?tab=api+keys."
                )

    @classmethod
    def create_session(cls, identifier: str, args: Optional[Dict[str, Any]] = None):
        """
        Create an authenticated HUBTrainingSession or return None.

        Args:
            identifier (str): Model identifier used to initialize the HUB training session.
            args (Dict[str, Any], optional): Arguments for creating a new model if identifier is not a HUB model URL.

        Returns:
            session (HUBTrainingSession | None): An authenticated session or None if creation fails.
        """
        try:
            session = cls(identifier)
            if args and not identifier.startswith(f"{HUB_WEB_ROOT}/models/"):  # not a HUB model URL
                session.create_model(args)
                assert session.model.id, "HUB model not loaded correctly"
            return session
        # PermissionError and ModuleNotFoundError indicate hub-sdk not installed
        except (PermissionError, ModuleNotFoundError, AssertionError):
            return None

    def load_model(self, model_id: str):
        """
        Load an existing model from Ultralytics HUB using the provided model identifier.

        Args:
            model_id (str): The identifier of the model to load.

        Raises:
            ValueError: If the specified HUB model does not exist.
        """
        self.model = self.client.model(model_id)
        if not self.model.data:  # then model does not exist
            raise ValueError(emojis("❌ The specified HUB model does not exist"))  # TODO: improve error handling

        self.model_url = f"{HUB_WEB_ROOT}/models/{self.model.id}"
        if self.model.is_trained():
            LOGGER.info(f"Loading trained HUB model {self.model_url} 🚀")
            url = self.model.get_weights_url("best")  # download URL with auth
            self.model_file = checks.check_file(url, download_dir=Path(SETTINGS["weights_dir"]) / "hub" / self.model.id)
            return

        # Set training args and start heartbeats for HUB to monitor agent
        self._set_train_args()
        self.model.start_heartbeat(self.rate_limits["heartbeat"])
        LOGGER.info(f"{PREFIX}View model at {self.model_url} 🚀")

    def create_model(self, model_args: Dict[str, Any]):
        """
        Initialize a HUB training session with the specified model arguments.

        Args:
            model_args (Dict[str, Any]): Arguments for creating the model, including batch size, epochs, image size,
                etc.

        Returns:
            (None): If the model could not be created.
        """
        payload = {
            "config": {
                "batchSize": model_args.get("batch", -1),
                "epochs": model_args.get("epochs", 300),
                "imageSize": model_args.get("imgsz", 640),
                "patience": model_args.get("patience", 100),
                "device": str(model_args.get("device", "")),  # convert None to string
                "cache": str(model_args.get("cache", "ram")),  # convert True, False, None to string
            },
            "dataset": {"name": model_args.get("data")},
            "lineage": {
                "architecture": {"name": self.filename.replace(".pt", "").replace(".yaml", "")},
                "parent": {},
            },
            "meta": {"name": self.filename},
        }

        if self.filename.endswith(".pt"):
            payload["lineage"]["parent"]["name"] = self.filename

        self.model.create_model(payload)

        # Model could not be created
        # TODO: improve error handling
        if not self.model.id:
            return None

        self.model_url = f"{HUB_WEB_ROOT}/models/{self.model.id}"

        # Start heartbeats for HUB to monitor agent
        self.model.start_heartbeat(self.rate_limits["heartbeat"])

        LOGGER.info(f"{PREFIX}View model at {self.model_url} 🚀")

    @staticmethod
    def _parse_identifier(identifier: str):
        """
        Parse the given identifier to determine the type and extract relevant components.

        The method supports different identifier formats:
            - A HUB model URL https://hub.ultralytics.com/models/MODEL
            - A HUB model URL with API Key https://hub.ultralytics.com/models/MODEL?api_key=APIKEY
            - A local filename that ends with '.pt' or '.yaml'

        Args:
            identifier (str): The identifier string to be parsed.

        Returns:
            api_key (str | None): Extracted API key if present.
            model_id (str | None): Extracted model ID if present.
            filename (str | None): Extracted filename if present.

        Raises:
            HUBModelError: If the identifier format is not recognized.
        """
        api_key, model_id, filename = None, None, None
        if str(identifier).endswith((".pt", ".yaml")):
            filename = identifier
        elif identifier.startswith(f"{HUB_WEB_ROOT}/models/"):
            parsed_url = urlparse(identifier)
            model_id = Path(parsed_url.path).stem  # handle possible final backslash robustly
            query_params = parse_qs(parsed_url.query)  # dictionary, i.e. {"api_key": ["API_KEY_HERE"]}
            api_key = query_params.get("api_key", [None])[0]
        else:
            raise HUBModelError(f"model='{identifier} invalid, correct format is {HUB_WEB_ROOT}/models/MODEL_ID")
        return api_key, model_id, filename

    def _set_train_args(self):
        """
        Initialize training arguments and create a model entry on the Ultralytics HUB.

        This method sets up training arguments based on the model's state and updates them with any additional
        arguments provided. It handles different states of the model, such as whether it's resumable, pretrained,
        or requires specific file setup.

        Raises:
            ValueError: If the model is already trained, if required dataset information is missing, or if there are
                issues with the provided training arguments.
        """
        if self.model.is_resumable():
            # Model has saved weights
            self.train_args = {"data": self.model.get_dataset_url(), "resume": True}
            self.model_file = self.model.get_weights_url("last")
        else:
            # Model has no saved weights
            self.train_args = self.model.data.get("train_args")  # new response

            # Set the model file as either a *.pt or *.yaml file
            self.model_file = (
                self.model.get_weights_url("parent") if self.model.is_pretrained() else self.model.get_architecture()
            )

        if "data" not in self.train_args:
            # RF bug - datasets are sometimes not exported
            raise ValueError("Dataset may still be processing. Please wait a minute and try again.")

        self.model_file = checks.check_yolov5u_filename(self.model_file, verbose=False)  # YOLOv5->YOLOv5u
        self.model_id = self.model.id

    def request_queue(
        self,
        request_func,
        retry: int = 3,
        timeout: int = 30,
        thread: bool = True,
        verbose: bool = True,
        progress_total: Optional[int] = None,
        stream_response: Optional[bool] = None,
        *args,
        **kwargs,
    ):
        """
        Execute request_func with retries, timeout handling, optional threading, and progress tracking.

        Args:
            request_func (callable): The function to execute.
            retry (int): Number of retry attempts.
            timeout (int): Maximum time to wait for the request to complete.
            thread (bool): Whether to run the request in a separate thread.
            verbose (bool): Whether to log detailed messages.
            progress_total (int, optional): Total size for progress tracking.
            stream_response (bool, optional): Whether to stream the response.
            *args (Any): Additional positional arguments for request_func.
            **kwargs (Any): Additional keyword arguments for request_func.

        Returns:
            (requests.Response | None): The response object if thread=False, otherwise None.
        """

        def retry_request():
            """Attempt to call request_func with retries, timeout, and optional threading."""
            t0 = time.time()  # Record the start time for the timeout
            response = None
            for i in range(retry + 1):
                if (time.time() - t0) > timeout:
                    LOGGER.warning(f"{PREFIX}Timeout for request reached. {HELP_MSG}")
                    break  # Timeout reached, exit loop

                response = request_func(*args, **kwargs)
                if response is None:
                    LOGGER.warning(f"{PREFIX}Received no response from the request. {HELP_MSG}")
                    time.sleep(2**i)  # Exponential backoff before retrying
                    continue  # Skip further processing and retry

                if progress_total:
                    self._show_upload_progress(progress_total, response)
                elif stream_response:
                    self._iterate_content(response)

                if HTTPStatus.OK <= response.status_code < HTTPStatus.MULTIPLE_CHOICES:
                    # if request related to metrics upload
                    if kwargs.get("metrics"):
                        self.metrics_upload_failed_queue = {}
                    return response  # Success, no need to retry

                if i == 0:
                    # Initial attempt, check status code and provide messages
                    message = self._get_failure_message(response, retry, timeout)

                    if verbose:
                        LOGGER.warning(f"{PREFIX}{message} {HELP_MSG} ({response.status_code})")

                if not self._should_retry(response.status_code):
                    LOGGER.warning(f"{PREFIX}Request failed. {HELP_MSG} ({response.status_code}")
                    break  # Not an error that should be retried, exit loop

                time.sleep(2**i)  # Exponential backoff for retries

            # if request related to metrics upload and exceed retries
            if response is None and kwargs.get("metrics"):
                self.metrics_upload_failed_queue.update(kwargs.get("metrics"))

            return response

        if thread:
            # Start a new thread to run the retry_request function
            threading.Thread(target=retry_request, daemon=True).start()
        else:
            # If running in the main thread, call retry_request directly
            return retry_request()

    @staticmethod
    def _should_retry(status_code: int) -> bool:
        """Determine if a request should be retried based on the HTTP status code."""
        retry_codes = {
            HTTPStatus.REQUEST_TIMEOUT,
            HTTPStatus.BAD_GATEWAY,
            HTTPStatus.GATEWAY_TIMEOUT,
        }
        return status_code in retry_codes

    def _get_failure_message(self, response: requests.Response, retry: int, timeout: int) -> str:
        """
        Generate a retry message based on the response status code.

        Args:
            response (requests.Response): The HTTP response object.
            retry (int): The number of retry attempts allowed.
            timeout (int): The maximum timeout duration.

        Returns:
            (str): The retry message.
        """
        if self._should_retry(response.status_code):
            return f"Retrying {retry}x for {timeout}s." if retry else ""
        elif response.status_code == HTTPStatus.TOO_MANY_REQUESTS:  # rate limit
            headers = response.headers
            return (
                f"Rate limit reached ({headers['X-RateLimit-Remaining']}/{headers['X-RateLimit-Limit']}). "
                f"Please retry after {headers['Retry-After']}s."
            )
        else:
            try:
                return response.json().get("message", "No JSON message.")
            except AttributeError:
                return "Unable to read JSON."

    def upload_metrics(self):
        """Upload model metrics to Ultralytics HUB."""
        return self.request_queue(self.model.upload_metrics, metrics=self.metrics_queue.copy(), thread=True)

    def upload_model(
        self,
        epoch: int,
        weights: str,
        is_best: bool = False,
        map: float = 0.0,
        final: bool = False,
    ) -> None:
        """
        Upload a model checkpoint to Ultralytics HUB.

        Args:
            epoch (int): The current training epoch.
            weights (str): Path to the model weights file.
            is_best (bool): Indicates if the current model is the best one so far.
            map (float): Mean average precision of the model.
            final (bool): Indicates if the model is the final model after training.
        """
        weights = Path(weights)
        if not weights.is_file():
            last = weights.with_name(f"last{weights.suffix}")
            if final and last.is_file():
                LOGGER.warning(
                    f"{PREFIX} Model 'best.pt' not found, copying 'last.pt' to 'best.pt' and uploading. "
                    "This often happens when resuming training in transient environments like Google Colab. "
                    "For more reliable training, consider using Ultralytics HUB Cloud. "
                    "Learn more at https://docs.ultralytics.com/hub/cloud-training."
                )
                shutil.copy(last, weights)  # copy last.pt to best.pt
            else:
                LOGGER.warning(f"{PREFIX} Model upload issue. Missing model {weights}.")
                return

        self.request_queue(
            self.model.upload_model,
            epoch=epoch,
            weights=str(weights),
            is_best=is_best,
            map=map,
            final=final,
            retry=10,
            timeout=3600,
            thread=not final,
            progress_total=weights.stat().st_size if final else None,  # only show progress if final
            stream_response=True,
        )

    @staticmethod
    def _show_upload_progress(content_length: int, response: requests.Response) -> None:
        """Display a progress bar to track the upload progress of a file download."""
        with TQDM(total=content_length, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            for data in response.iter_content(chunk_size=1024):
                pbar.update(len(data))

    @staticmethod
    def _iterate_content(response: requests.Response) -> None:
        """Process the streamed HTTP response data."""
        for _ in response.iter_content(chunk_size=1024):
            pass  # Do nothing with data chunks
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
