import re


class Reranker:

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:

        query_words = self._tokenize(query)

        scored_chunks = []

        for chunk in chunks:

            chunk_words = self._tokenize(
                chunk["text"]
            )

            if not query_words:
                score = 0.0
            else:
                matched_words = (
                    query_words & chunk_words
                )

                score = (
                    len(matched_words)
                    / len(query_words)
                )

            chunk["rerank_score"] = score

            scored_chunks.append(chunk)

        scored_chunks.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return scored_chunks[:top_k]

    def _tokenize(self, text: str) -> set[str]:

        words = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )

        return set(words)