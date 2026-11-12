class LayaModel:
    def __init__(self, source: str, device: str):
        import laya

        self.agent = laya.load(source, device=device)

    def choose(self, transcript: str, criteria: dict):
        prediction = self.agent.predict(
            transcript,
            {"route": {
                "type": "choice",
                "instructions": "Chọn cách xử lý yêu cầu bằng giọng nói của người dùng.",
                "criteria": criteria,
            }},
        )
        answer = prediction["answers"]["route"]
        return answer["choice"], float(answer["answer_confidence"])
