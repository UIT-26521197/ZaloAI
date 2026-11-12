from ..core.types import Route

ROUTES = {
    "direct": "Đã đủ thông tin để gọi công cụ phù hợp trực tiếp.",
    "lookup": "Cần tra cứu dữ liệu hoặc thực hiện nhiều bước phụ thuộc nhau.",
    "clarify": "Thiếu chi tiết quan trọng, có thể cần hỏi lại.",
    "general": "Yêu cầu chưa rõ, có nhiều cách hiểu hoặc nằm ngoài phạm vi công cụ.",
}


def route_request(laya, transcript: str, threshold: float) -> Route:
    choice, confidence = laya.choose(transcript, ROUTES)
    if choice not in ROUTES or not threshold <= confidence <= 1.0:
        return Route("general", confidence if 0 <= confidence <= 1 else None)
    return Route(choice, confidence)
