"""Chuẩn hoá văn bản (ghi lại đầy đủ theo yêu cầu tái lập).

Các bước, theo thứ tự:
1. html.unescape (&amp; ... -> ký tự thật)
2. Thẻ <br> (mọi biến thể) -> khoảng trắng
3. Chuẩn hoá Unicode NFC (gộp dấu tiếng Việt về dạng dựng sẵn)
4. Gộp khoảng trắng liên tiếp, strip
Giữ nguyên emoji, dấu câu và hoa/thường (hạ chữ thường chỉ làm ở bước vector hoá khi cần).
KHÔNG sửa teencode hay kéo dài ký tự ở cấu hình chính.
"""
import html
import re
import unicodedata

_BR = re.compile(r"<\s*br\s*/?\s*>", re.I)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = html.unescape(str(text))
    text = _BR.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    return _WS.sub(" ", text).strip()
