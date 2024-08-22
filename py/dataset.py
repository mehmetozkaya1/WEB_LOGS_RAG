# Import necessary libraries
import re
from tqdm.auto import tqdm
import pandas as pd
import random

# Global variables
ACCESS_LOG_FILE_PATH = "../data/access.log" # Our access data path
LOG_PATTERN = re.compile( # Log pattern in our access.log data
    r'(?P<ip>\S+) - - \[(?P<timestamp>.*?)\] "(?P<method>\S+) (?P<url>\S+) '
    r'(?P<protocol>\S+)" (?P<status>\d{3}) (?P<size>\d+) "(?P<referrer>.*?)" '
    r'"(?P<user_agent>.*?)" "(?P<other>.*?)"'
)
DELETED_KEYS = ["url" ,"embedding", "referrer", "other"] # (Optional)
NUM_EXAMPLES = 1000 # Take the first 1000 examples

def read_access_log_file(access_log_file_path, log_pattern, num_examples=1000):
    """
    A function to read access log file and convert it into a dictionary.
    """
    access_logs_dict = []
    log_pattern = re.compile(log_pattern)  # log_pattern'ı derleyin
    with open(access_log_file_path, "r") as file:
        for _ in tqdm(range(num_examples), desc="Processing lines"):
            line = file.readline()
            if not line:
                break
            match = log_pattern.match(line)
            if match:
                log_data = match.groupdict()
                access_logs_dict.append(log_data)

    return access_logs_dict

def drop_keys(access_logs_dict, deleted_keys):
    """
    A function to delete unnecessary keys in the dictionary
    """
    for key in deleted_keys:
        for idx in tqdm(range(len(access_logs_dict))):
            del access_logs_dict[idx][key]

def extract_browser_and_os(user_agent):
    """
    A function to extrack browser and os name from url
    """
    browser_match = re.search(r'\b(Chrome|Firefox|Safari|Opera|Edge|MSIE|Trident|Googlebot|bing|AhrefsBot)\b', user_agent)
    if browser_match:
        browser_info = browser_match.group(0)
        if browser_info in ["MSIE", "Trident"]:
            browser_info = "Internet Explorer"
    else:
        browser_info = "unknown"

    os_match = re.search(r'\b(Windows NT|Android|iPhone|iPad|Mac OS X|Linux|Windows Phone|Macintosh)\b', user_agent)
    if os_match:
        os_info = os_match.group(0)
        # İşletim sistemi açıklamalarını daha okunabilir hale getir
        if "Windows NT" in os_info:
            windows_version = re.search(r'Windows NT (\d+\.\d+)', user_agent)
            if windows_version:
                version = windows_version.group(1)
                os_info = f"Windows {version}"
            else:
                os_info = "Windows"
        elif os_info == "Mac OS X":
            os_info = "Mac OS"
    else:
        os_info = "unknown OS"

    return browser_info, os_info

def convert_to_context(access_logs_dict):
    """
    A function to create our context using log data.
    """
    for idx, item in enumerate(access_logs_dict):
        url = item.get('url', '')
        user_agent = item.get('user_agent', '')

        is_mobile = any(keyword in user_agent for keyword in ["Mobile", "Android", "iPhone", "iPad", "Windows Phone"])

        if '/static/images/' in url:
            if "amp" in url:
                if 'blog.png' in url:
                    action = "accessed a blog image"
                elif 'instagram.png' in url:
                    action = "accessed an Instagram image"
                elif 'telegram.png' in url:
                    action = "accessed a Telegram image"
            elif 'guarantees/' in url:
                guarantee_type = url.split('/static/images/guarantees/')[1].split('.')[0]
                guarantee_type = guarantee_type.replace('-', ' ')
                action = f"viewed a guarantee image for {guarantee_type}"
            else:
                action = "loaded a static image"
        elif '/static/' in url:
            if 'css' in url:
                action = "accessed a CSS file"
            elif 'js' in url:
                action = "accessed a JavaScript file"
            elif 'png' in url:
                action = "accessed an image"
        elif '/image/' in url:
            product_id_match = re.search(r'/image/(\d+)', url)
            product_id = product_id_match.group(1) if product_id_match else "unknown"
            action = f"viewed a product image with ID {product_id}"
        elif '/product/' in url:
            product_id = url.split('/product/')[1].split('/')[0]
            action = f"viewed product with ID {product_id}"
        elif '/filter/' in url:
            filter_params = url.split('/filter/')[1]
            filter_params = filter_params.replace('%2C', ',')
            action = f"applied filter parameters: {filter_params}"
        elif '/m/product/' in url:
            product_id = url.split('/m/product/')[1].split('/')[0]
            action = f"viewed product with ID {product_id}"
        elif '/m/filter/' in url:
            filter_params = url.split('/m/filter/')[1]
            filter_params = filter_params.replace('%2C', ',')
            action = f"applied filter parameters: {filter_params}"
        elif '/settings/logo' in url:
            action = "accessed logo settings"
        elif '/m/article/' in url:
            article_id = url.split('/m/article/')[1]
            action = f"viewed article with ID {article_id}"
        elif '/m/browse/' in url:
            browse_term = url.split('/m/browse/')[1].replace('-', ' ')
            action = f"searched for {browse_term}"
        elif '/ajaxFilter/' in url:
            filter_params = url.split('/ajaxFilter/')[1].split('?')[0]
            page_number = re.search(r'page=(\d+)', url)
            page_number = page_number.group(1) if page_number else "unknown"
            action = f"applied filter parameters {filter_params} on page {page_number}"
        elif '/m/updateVariation' in url:
            action = "updated product variation"
        elif 'site/ping' in url:
            action = "pinged the site"
        elif '/search' in url:
            action = "searched a word in the site"
        else:
            action = "accessed a page"

        if item['status'] == '200':
            status_info = "successfully accessed the page"
        elif item['status'] == '404':
            status_info = "encountered an error while accessing the page"
        elif item['status'] == '302':
            status_info = "found a result in the page"
        elif item['status'] == '301':
            status_info = "the page user tried to access was moved permanently"
        else:
            status_info = "unknown status"

        device_info = "on a mobile device" if is_mobile else "on a desktop device"
        browser_info, os_info = extract_browser_and_os(user_agent)
        user_agent_info = f"using {browser_info} on {os_info}"

        context = (
            f"Using the {item.get('method', 'unspecified')} method, "
            f"{status_info}. The user {action} {device_info} and was {user_agent_info}."
        )

        item['log_index'] = idx
        item["context"] = context

def dict_to_df(access_logs_dict):
    """
    A function to convert dictionary into a Pandas DataFrame
    """
    df = pd.DataFrame(access_logs_dict)

    return df

# Read the log file and process it
access_logs_dict = list(read_access_log_file(ACCESS_LOG_FILE_PATH, LOG_PATTERN, NUM_EXAMPLES))
convert_to_context(access_logs_dict)
access_logs_dict_df = dict_to_df(access_logs_dict)
print(random.sample(access_logs_dict, k = 2))

# Optionally drop unnecessary keys
# drop_keys(access_logs_dict, DELETED_KEYS)