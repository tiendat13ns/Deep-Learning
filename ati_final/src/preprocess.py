"""Chuẩn hoá văn bản (ghi lại đầy đủ theo yêu cầu tái lập).

Các bước, theo thứ tự:
1. html.unescape (&amp; ... -> ký tự thật)
2. Thẻ <br> (mọi biến thể) -> khoảng trắng
3. Chuẩn hoá Unicode NFC (gộp dấu tiếng Việt về dạng dựng sẵn)
4. Gộp khoảng trắng liên tiếp, strip
Giữ nguyên emoji, dấu câu và hoa/thường (hạ chữ thường chỉ làm ở bước vector hoá khi cần).
KHÔNG sửa teencode hay kéo dài ký tự ở cấu hình chính.

Biến thể riêng cho PhoBERT (emoji_to_text): tokenizer của PhoBERT ánh xạ MỌI emoji về <unk>, nên có thể thay
emoji bằng tên tiếng Anh của nó (thư viện `emoji`, ví dụ "loudly crying face") trước khi tách từ.
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


def emoji_to_text(text: str) -> str:
    """Thay mỗi emoji bằng tên của nó (gạch dưới -> khoảng trắng); dùng khi mô hình không có emoji trong từ vựng."""
    import emoji

    text = emoji.demojize(text, delimiters=(" ", " ")).replace("_", " ").replace("️", "")
    return _WS.sub(" ", text).strip()
