from bs4 import BeautifulSoup
import html2text
import requests
import re
import time
import os
import random
from urllib.parse import urlparse

def extract_clean_body(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')

    # 删除 <script> 和 <style> 标签
    for tag in soup(['script', 'style']):
        tag.decompose()

    # 获取 <body> 标签
    body = soup.body
    if not body:
        return ""

    # 清除所有标签的属性
    for tag in body.find_all(True):
        tag.attrs = {}

    # 获取 <body> 的 HTML 内容
    clean_html = body.decode_contents()

    # 去除多余换行和空格，压缩成单行字符串
    compressed_html = re.sub(r'\s+', ' ', clean_html).strip()

    compressed_html = "<body>" + compressed_html + "</body>"
    return compressed_html


def html_to_markdown(html: str) -> str:
    return html2text.html2text(html)

def main():
    urls = [
        # "https://beautiful-soup-4.readthedocs.io/en/latest/",
        # "https://www.runoob.com/python3/python-spider-beautifulsoup.html",
        # "https://www.bilibili.com/opus/725617098290626568?from=search&spm_id_from=333.337.0.0",
        # "https://blog.csdn.net/qq_26169815/article/details/149334725?spm=1001.2100.3001.7377&utm_medium=distribute.pc_feed_blog.none-task-blog-personrec_tag-1-149334725-null-null.nonecase&depth_1-utm_source=distribute.pc_feed_blog.none-task-blog-personrec_tag-1-149334725-null-null.nonecase",
        # "https://zh.wikipedia.org/wiki/%E6%89%AC%E5%B0%BC%E5%85%8B%C2%B7%E8%BE%9B%E7%BA%B3",
        # "https://blog.csdn.net/qq_43350524/article/details/149281863?spm=1001.2100.3001.7377&utm_medium=distribute.pc_feed_blog.none-task-blog-hot-3-149281863-null-null.nonecase&depth_1-utm_source=distribute.pc_feed_blog.none-task-blog-hot-3-149281863-null-null.nonecase",
        # "https://blog.csdn.net/qq_43350524/article/details/145886801",
        # "https://blog.csdn.net/m0_74087691/article/details/149244512",
        # "https://www.cnblogs.com/zh-dream/p/12834056.html",
        # "https://www.cnblogs.com/kqdssheng/p/18985420",
        # "https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Reference/Headers/Host",
        # "https://www.jianshu.com/p/424e037c5dd8",
        "https://cloud.tencent.com/developer/article/2014168",
        "https://www.bilibili.com/opus/1044217170887704584",
        "https://blog.csdn.net/Triste__chengxi/article/details/149183904"
    ]
    save_base_folder = "data/sources"
    for url in urls:
        print(f"crawling {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        }
        try:
            file_name = f"{int(time.time() / 60)}_{random.randint(1000, 9999)}.md"
            parsed_url = urlparse(url)
            headers["Host"] = parsed_url.hostname
            save_to = os.path.join(save_base_folder, file_name)
            response = requests.get(url,headers=headers)
            html = response.text
            clean_html = extract_clean_body(html)
            markdown = html_to_markdown(clean_html)
            with open(save_to, "w", encoding="utf-8") as f:
                f.write(markdown)
        except Exception as e:
            print(f"[ERROR] Failed to fetch {url}: {e}")
            continue

if __name__ == "__main__":
    main()