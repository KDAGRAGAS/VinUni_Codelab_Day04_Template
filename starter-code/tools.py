import json
import os
import re
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta

# Luôn tính đường dẫn dựa trên vị trí file này (tránh Trap 1: FileNotFoundError
# khi chạy pytest / python từ một thư mục khác).
RAW_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raw-data"))

VALID_CATEGORIES = ("xe_dien", "du_lich")
VALID_PRIORITIES = ("low", "medium", "high")
VN_TZ = timezone(timedelta(hours=7))  # Giờ Việt Nam (UTC+07:00)


def _load_json_list(path: str) -> List[Dict[str, Any]]:
    """Đọc file JSON dạng list. Ném ValueError nếu file hỏng / sai định dạng."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    data = json.loads(content)
    if not isinstance(data, list):
        raise ValueError(f"{os.path.basename(path)} phải chứa một JSON list.")
    return data


def _normalize_category(category: str) -> str:
    """'Xe Điện ', 'xe-dien' → 'xe_dien' để so khớp không phân biệt hoa thường / dấu cách."""
    return re.sub(r"[\s\-]+", "_", str(category).strip().lower())


# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.

    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.

    Returns:
        Danh sách sản phẩm phù hợp điều kiện (sắp xếp theo giá tăng dần).
        Trả về list rỗng nếu không có sản phẩm nào khớp.
        Trả về [{"error": ...}] nếu file dữ liệu thiếu hoặc hỏng.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]

    try:
        products = _load_json_list(catalog_file)
    except (json.JSONDecodeError, ValueError) as e:
        return [{"error": f"Không đọc được product catalog: {e}"}]

    # LLM đôi khi truyền max_price dạng string ("600000000") hoặc None → chuẩn hoá.
    if max_price is None:
        max_price = 999999999999
    try:
        max_price = int(max_price)
    except (TypeError, ValueError):
        return [{"error": f"max_price không hợp lệ: {max_price!r}. Phải là số nguyên (VNĐ)."}]

    wanted = _normalize_category(category)

    results = [
        p for p in products
        if _normalize_category(p.get("category", "")) == wanted
        and isinstance(p.get("price_vnd"), (int, float))
        and p["price_vnd"] <= max_price
    ]
    results.sort(key=lambda p: p["price_vnd"])
    return results


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# ---------------------------------------------------------------------------

def _next_sequence(tickets: List[Dict[str, Any]]) -> int:
    """
    Số thứ tự kế tiếp = max(seq hiện có) + 1.
    An toàn hơn len()+1: không sinh trùng ID nếu có ticket bị xoá khỏi file.
    """
    max_seq = 0
    for t in tickets:
        match = re.match(r"^TK-\d{8}-(\d+)$", str(t.get("ticket_id", "")))
        if match:
            max_seq = max(max_seq, int(match.group(1)))
    return max(max_seq, len(tickets)) + 1


def _atomic_write_json(path: str, data: Any) -> None:
    """Ghi ra file tạm rồi thay thế → file gốc không bị hỏng nếu ghi giữa chừng thất bại."""
    directory = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.

    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.

    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
        Trả về {"error": ...} nếu đầu vào không hợp lệ hoặc không ghi được file.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")

    # --- Validate input ----------------------------------------------------
    customer_name = (customer_name or "").strip()
    issue_description = (issue_description or "").strip()
    if not customer_name:
        return {"error": "Thiếu customer_name — cần hỏi tên khách hàng trước khi tạo ticket."}
    if not issue_description:
        return {"error": "Thiếu issue_description — cần mô tả vấn đề trước khi tạo ticket."}

    priority = (priority or "medium").strip().lower()
    if priority not in VALID_PRIORITIES:
        priority = "medium"

    # --- Load existing tickets (tránh Trap 2: ghi đè thay vì append) -------
    existing_tickets: List[Dict[str, Any]] = []
    if os.path.exists(tickets_file):
        try:
            existing_tickets = _load_json_list(tickets_file)
        except (json.JSONDecodeError, ValueError) as e:
            # KHÔNG ghi đè file hỏng — sẽ làm mất toàn bộ ticket cũ.
            return {"error": f"File support_tickets.json bị lỗi, không thể tạo ticket: {e}"}

    # --- Generate ticket ID ------------------------------------------------
    now = datetime.now(VN_TZ)
    today = now.strftime("%Y%m%d")
    seq = _next_sequence(existing_tickets)
    ticket_id = f"TK-{today}-{seq:03d}"

    # --- Create & save new ticket ------------------------------------------
    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": now.isoformat(),  # vd: 2026-11-09T17:30:00.123456+07:00
        "category": "general"
    }
    existing_tickets.append(new_ticket)

    try:
        _atomic_write_json(tickets_file, existing_tickets)
    except OSError as e:
        return {"error": f"Không lưu được ticket: {e}"}

    return {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "priority": priority,
        "status": "open",
        "message": f"Ticket {ticket_id} đã được tạo thành công."
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": (
            "Tra cứu sản phẩm/dịch vụ Vingroup (xe điện VinFast, gói du lịch/nghỉ dưỡng Vinpearl) "
            "theo danh mục và giá tối đa. Trả về danh sách sản phẩm gồm tên, giá (VNĐ), mô tả, "
            "tính năng, tình trạng hàng. PHẢI gọi tool này mỗi khi khách hỏi về sản phẩm, giá "
            "hoặc tư vấn mua — không được tự bịa thông tin. Trả về list rỗng nếu không có sản phẩm phù hợp."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Loại sản phẩm: 'xe_dien' cho xe điện VinFast, "
                        "'du_lich' cho resort/khách sạn/tour Vinpearl."
                    ),
                    "enum": list(VALID_CATEGORIES)
                },
                "max_price": {
                    "type": "integer",
                    "description": (
                        "Giá tối đa tính bằng VNĐ (số nguyên, không dấu phân cách). "
                        "Ví dụ: 'dưới 600 triệu' → 600000000; 'dưới 6 triệu' → 6000000. "
                        "Bỏ qua nếu khách không nêu ngân sách."
                    ),
                    "minimum": 0
                }
            },
            "required": ["category"]
        }
    },
    {
        "name": "submit_support_ticket",
        "description": (
            "Tạo ticket hỗ trợ khi khách hàng báo lỗi, khiếu nại, phản hồi về sản phẩm/dịch vụ "
            "Vingroup hoặc cần nhân viên xử lý. Trả về ticket_id (dạng TK-YYYYMMDD-NNN) và trạng thái. "
            "Chỉ gọi khi đã có tên khách hàng và mô tả vấn đề; nếu thiếu, hãy hỏi lại khách trước."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Họ tên đầy đủ của khách hàng, ví dụ: 'Lê Minh Khoa'."
                },
                "issue_description": {
                    "type": "string",
                    "description": (
                        "Mô tả ngắn gọn, cụ thể vấn đề khách gặp phải, "
                        "ví dụ: 'Xe VF 8 bị lỗi hệ thống ADAS'."
                    )
                },
                "priority": {
                    "type": "string",
                    "description": (
                        "Mức độ ưu tiên. 'high' khi khách nói nghiêm trọng/gấp/khẩn cấp hoặc ảnh hưởng an toàn; "
                        "'medium' khi mức độ trung bình hoặc không nói rõ; 'low' khi chỉ là góp ý nhỏ."
                    ),
                    "enum": list(VALID_PRIORITIES),
                    "default": "medium"
                }
            },
            "required": ["customer_name", "issue_description"]
        }
    }
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
