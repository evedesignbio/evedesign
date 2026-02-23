import time
import requests
from loguru import logger

def _request_with_retries(
    method,
    url,
    *,
    data=None,
    params=None,
    headers=None,
    timeout=6.02,
    max_retries=5,
    retry_sleep=5,
    context="server",
    raise_for_status=True,
):
    error_count = 0
    while True:
        try:
            response = requests.request(
                method,
                url,
                data=data,
                params=params,
                timeout=timeout,
                headers=headers,
            )
            if raise_for_status:
                response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            logger.warning("Timeout while contacting %s. Retrying...", context)
            continue
        except Exception as exc:
            error_count += 1
            logger.warning(
                "Error while contacting %s. Retrying... (%d/%d)",
                context,
                error_count,
                max_retries,
            )
            logger.warning("Error: %s", exc)
            time.sleep(retry_sleep)
            if error_count >= max_retries:
                raise
