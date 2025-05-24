import re
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

class PDFChatbot:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_chunks = []
        self.chunk_labels = []
        self.embeddings = None
        self.nn = None
        self.feedback_data = []

    def chunk_text(self, text, max_chunk_size=500):
        sentences = text.replace('\n', ' ').split('.')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence += '.'

            if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def label_chunk(self, chunk):
        chunk_lower = chunk.lower()
        if re.search(r'\babstract\b', chunk_lower):
            return 'abstract'
        elif re.search(r'\b(introduction|background)\b', chunk_lower):
            return 'introduction'
        elif re.search(r'\b(conclusion|summary|concluding remarks)\b', chunk_lower):
            return 'conclusion'
        else:
            return 'general'

    def ingest_text(self, text, max_chunk_size=500):
        self.text_chunks = self.chunk_text(text, max_chunk_size)
        self.chunk_labels = [self.label_chunk(chunk) for chunk in self.text_chunks]
        self.embeddings = self.model.encode(self.text_chunks, convert_to_numpy=True)
        self.nn = NearestNeighbors(n_neighbors=3, metric='cosine')
        self.nn.fit(self.embeddings)

    def dynamic_threshold(self, query):
        length = len(query.split())
        if length <= 3:
            return 0.05
        elif length <= 7:
            return 0.1
        else:
            return 0.15

    def get_response(self, query, top_k=3):
        if not self.nn:
            return "No document ingested yet."

        query_lower = query.lower()
        similarity_threshold = self.dynamic_threshold(query)

        # Check for labeled sections
        target_label = None
        if 'abstract' in query_lower:
            target_label = 'abstract'
        elif 'introduction' in query_lower or 'background' in query_lower:
            target_label = 'introduction'
        elif 'conclusion' in query_lower or 'summary' in query_lower:
            target_label = 'conclusion'

        if target_label:
            indices_to_search = [i for i, label in enumerate(self.chunk_labels) if label == target_label]
            if not indices_to_search:
                return f"Sorry, this document doesn't have a clearly labeled {target_label} section."

            subset_embeddings = self.embeddings[indices_to_search]
            query_vec = self.model.encode([query])

            nn_subset = NearestNeighbors(n_neighbors=min(top_k, len(indices_to_search)), metric='cosine')
            nn_subset.fit(subset_embeddings)
            distances, indices = nn_subset.kneighbors(query_vec, n_neighbors=min(top_k, len(indices_to_search)))

            responses = []
            for dist, idx in zip(distances[0], indices[0]):
                similarity = 1 - dist
                if similarity >= similarity_threshold:
                    responses.append(self.text_chunks[indices_to_search[idx]])

            if not responses:
                return f"Sorry, I couldn't find relevant information in the {target_label} section."

            # Return raw chunks concatenated as answer (no GPT summarization)
            return "\n\n".join(responses)

        else:
            query_vec = self.model.encode([query])
            distances, indices = self.nn.kneighbors(query_vec, n_neighbors=top_k)

            responses = []
            for dist, idx in zip(distances[0], indices[0]):
                similarity = 1 - dist
                if similarity >= similarity_threshold:
                    responses.append(self.text_chunks[idx])

            if not responses:
                return "Sorry, I couldn't find relevant information in the PDF."

            return "\n\n".join(responses)

    def add_feedback(self, query, response, feedback):
        self.feedback_data.append({
            "query": query,
            "response": response,
            "feedback": feedback
        })
